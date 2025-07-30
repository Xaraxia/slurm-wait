#!/usr/bin/env python3
"""
Slurm Wait Time Data Collector
------------------------------
Collects data about job wait times due to insufficient hardware rather than
policy restrictions. Stores results in a SQLite database for later analysis.

# Run with default settings (collect data every 5 minutes)
python slurm-data-collector.py

# Customize collection interval and database location
python slurm-data-collector.py --interval 900 --db /path/to/database.db

# Run for a specific duration (e.g., 24 hours)
python slurm-data-collector.py --max-runtime 86400
"""

import subprocess
import sqlite3
import time
import json
import traceback
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("slurm_collector.log"), logging.StreamHandler()],
)
logger = logging.getLogger("slurm_collector")

# Resource limitation reason codes - jobs with these reasons are excluded from our analysis
POLICY_LIMITATION_REASONS = [
    "QOSMaxCpuPerUserLimit",
    "QOSMaxJobsPerUserLimit",
    "QOSMaxGRESPerUser",
    "QOSMaxMemoryPerUser",
    "AssocGrpCPURunMinutesLimit",
    "AssocGrpNodeLimit",
    "AssocGrpMemLimit",
    "AssocGrpCpuLimit",
    "AssocGrpJobLimit",
    "AssocGrpSubmitJobLimit",
    "QOSGrpCpuLimit",
    "QOSGrpMemLimit",
    "QOSGrpNodeLimit",
    "QOSGrpJobLimit",
    "QOSGrpSubmitJobLimit",
    "Licenses",
    "PartitionTimeLimit",
    "AccountingPolicy",
    "Dependency",
]

# Hardware limit reason codes - jobs with these reasons are waiting due to insufficient hardware
HARDWARE_LIMITATION_REASONS = [
    "Resources",
    "Priority",
    "NodeDown",
    "ReqNodeNotAvail",
    "PartitionNodeLimit",
]


def create_database(db_path):
    """Create the SQLite database and tables if they don't exist"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for pending jobs
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS pending_jobs (
        timestamp INTEGER,
        job_id TEXT,
        partition TEXT,
        job_name TEXT,
        user TEXT,
        wait_time INTEGER,  -- in seconds
        requested_nodes INTEGER,
        requested_cpus INTEGER,
        requested_mem INTEGER,  -- in MB
        reason TEXT,
        is_hardware_limited BOOLEAN,
        tres_per_job TEXT  -- JSON string with TRES details
    )
    """
    )

    # Create table for cluster stats
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS cluster_stats (
        timestamp INTEGER,
        total_nodes INTEGER,
        available_nodes INTEGER,
        total_cpus INTEGER,
        allocated_cpus INTEGER,
        idle_cpus INTEGER,
        down_cpus INTEGER,
        total_memory INTEGER,  -- in MB
        allocated_memory INTEGER,  -- in MB
        partition_data TEXT  -- JSON string with per-partition details
    )
    """
    )

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")


def parse_memory_to_mb(mem_str):
    """Convert Slurm memory format to MB"""
    if not mem_str or mem_str == "N/A":
        return 0

    # Remove any non-alphanumeric characters
    mem_str = "".join(c for c in mem_str if c.isalnum())

    # Extract number and unit
    if mem_str[-1].isalpha():
        number = int(mem_str[:-1])
        unit = mem_str[-1].upper()
    else:
        return int(mem_str)  # Assume MB if no unit

    # Convert to MB
    if unit == "K":
        return number // 1024
    elif unit == "M":
        return number
    elif unit == "G":
        return number * 1024
    elif unit == "T":
        return number * 1024 * 1024
    else:
        return number  # Default to MB


def get_pending_jobs():
    """Get data about pending jobs from squeue --json including TRES-per-job"""
    try:
        # Use JSON format to get clean, parseable data
        cmd = ["squeue", "-t", "PD", "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse JSON output
        try:
            job_data = json.loads(result.stdout)

            # In Slurm's JSON output, the jobs are typically in a 'jobs' key
            if "jobs" in job_data:
                jobs_list = job_data["jobs"]
            else:
                jobs_list = job_data  # Fallback if structure is different

            logger.debug(
                f"First job structure: {json.dumps(jobs_list[0] if jobs_list else {})}"
            )
            current_time = time.time()
            jobs = []

            for job in jobs_list:
                # Extract job details
                job_id = str(job.get("job_id", ""))
                partition = job.get("partition", "")
                job_name = job.get("name", "")
                user = job.get("user_name", "")
                submit_time = job.get("submit_time", 0)
                nodes = job.get("min_nodes", 0)
                cpus = job.get("min_cpus", 0)
                mem = job.get("min_memory_per_node", "0")
                reason = job.get("state_reason", "")

                # Handle the case where submit_time is a dictionary
                if isinstance(submit_time, dict) and "number" in submit_time:
                    submit_time = submit_time["number"]
                elif isinstance(submit_time, dict):
                    logger.warning(
                        f"Unexpected submit_time structure for job {job_id}: {submit_time}"
                    )
                    submit_time = 0
                elif not isinstance(submit_time, (int, float)):
                    logger.warning(
                        f"Invalid submit_time format for job {job_id}: {submit_time}"
                    )
                    submit_time = 0

                # Get TRES info directly from JSON
                tres_data = {}
                if "tres_per_job" in job:
                    tres_data = job["tres_per_job"]
                elif "tres_req_str" in job:
                    # Parse TRES request string if available
                    tres_str = job["tres_req_str"]
                    parts = tres_str.split(",")
                    for part in parts:
                        if "=" in part:
                            key, value = part.split("=", 1)
                            key = key.strip()

                            # Handle numeric values
                            if value.isdigit():
                                tres_data[key] = int(value)
                            # Handle memory values with units
                            elif key == "mem" and any(
                                value.endswith(unit) for unit in ["K", "M", "G", "T"]
                            ):
                                tres_data[key] = parse_memory_to_mb(value)
                            else:
                                tres_data[key] = value

                # Determine if job is hardware-limited
                is_hardware_limited = False
                for hw_reason in HARDWARE_LIMITATION_REASONS:
                    if hw_reason in reason:
                        is_hardware_limited = True
                        break

                # Skip jobs that are limited by policy
                is_policy_limited = False
                for policy_reason in POLICY_LIMITATION_REASONS:
                    if policy_reason in reason:
                        is_policy_limited = True
                        break

                if is_policy_limited:
                    continue

                # Calculate wait time in seconds
                wait_seconds = int(current_time - submit_time) if submit_time else 0

                # Convert memory to MB if it's a string with units
                if isinstance(mem, str):
                    mem_mb = parse_memory_to_mb(mem)
                else:
                    mem_mb = mem

                jobs.append(
                    {
                        "job_id": job_id,
                        "partition": partition,
                        "job_name": job_name,
                        "user": user,
                        "wait_time": wait_seconds,
                        "submission_time": (
                            datetime.fromtimestamp(submit_time).isoformat()
                            if submit_time
                            else ""
                        ),
                        "requested_nodes": nodes,
                        "requested_cpus": cpus,
                        "requested_mem": mem_mb,
                        "reason": reason,
                        "is_hardware_limited": is_hardware_limited,
                        "tres_per_job": json.dumps(tres_data),
                    }
                )

            return jobs

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from squeue: {e}")
            return []

    except subprocess.SubprocessError as e:
        logger.error(f"Error getting pending jobs: {e}")

        # Check if squeue --json is not supported in this Slurm version
        if "unknown option" in str(e) and "--json" in str(e):
            logger.error(
                "The --json option is not supported in this version of Slurm. \
                    Please update Slurm or use the text-based version."
            )

        return []


def get_cluster_stats():
    """Get cluster statistics from sinfo"""
    try:
        node_cmd = ["sinfo", "-o", "%n|%c|%O|%m|%T|%P", "--noheader"]
        node_result = subprocess.run(
            node_cmd, capture_output=True, text=True, check=True
        )

        nodes = {}
        partitions = {}

        for line in node_result.stdout.strip().split("\n"):
            if not line:
                continue

            fields = line.split("|")
            if len(fields) < 6:
                continue

            node_name, cpus, cpu_load, memory, state, partition = fields

            # Process node data
            cpus = int(cpus) if cpus.isdigit() else 0
            memory = parse_memory_to_mb(memory)

            # Track node state
            if node_name not in nodes:
                nodes[node_name] = {
                    "cpus": cpus,
                    "memory": memory,
                    "state": state,
                    "partitions": [],
                }

            nodes[node_name]["partitions"].append(partition)

            # Track partition data
            for part in partition.split(","):
                if part not in partitions:
                    partitions[part] = {
                        "total_nodes": 0,
                        "available_nodes": 0,
                        "total_cpus": 0,
                        "available_cpus": 0,
                        "total_memory": 0,
                        "available_memory": 0,
                    }

                partitions[part]["total_nodes"] += 1
                partitions[part]["total_cpus"] += cpus
                partitions[part]["total_memory"] += memory

                if state in ["idle", "mix"]:
                    partitions[part]["available_nodes"] += 1
                    # For mixed nodes, count half CPUs as available (approximation)
                    if state == "mix":
                        partitions[part]["available_cpus"] += cpus // 2
                        partitions[part]["available_memory"] += memory // 2
                    else:
                        partitions[part]["available_cpus"] += cpus
                        partitions[part]["available_memory"] += memory

        # Calculate cluster-wide statistics
        total_nodes = len(nodes)
        available_nodes = sum(
            1
            for node in nodes.values()
            if "idle" in node["state"] or "mix" in node["state"]
        )
        total_cpus = sum(node["cpus"] for node in nodes.values())
        allocated_cpus = 0
        idle_cpus = 0
        down_cpus = 0
        total_memory = sum(node["memory"] for node in nodes.values())
        allocated_memory = 0

        for node in nodes.values():
            state = node["state"]
            if "allocated" in state:
                allocated_cpus += node["cpus"]
                allocated_memory += node["memory"]
            elif "idle" in state:
                idle_cpus += node["cpus"]
            elif "mix" in state:
                # Approximation for mixed nodes
                allocated_cpus += node["cpus"] // 2
                allocated_memory += node["memory"] // 2
                idle_cpus += node["cpus"] // 2
            elif any(s in state for s in ["down", "drain", "maint"]):
                down_cpus += node["cpus"]

        return {
            "total_nodes": total_nodes,
            "available_nodes": available_nodes,
            "total_cpus": total_cpus,
            "allocated_cpus": allocated_cpus,
            "idle_cpus": idle_cpus,
            "down_cpus": down_cpus,
            "total_memory": total_memory,
            "allocated_memory": allocated_memory,
            "partition_data": json.dumps(partitions),
        }
    except Exception as e:
        logger.error(f"Error getting cluster stats: {e}")
        logger.error(traceback.format_exc())
        return None


def store_data(db_path, timestamp, pending_jobs, cluster_stats):
    """Store collected data in the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Store pending jobs
        for job in pending_jobs:
            cursor.execute(
                """
            INSERT INTO pending_jobs (
                timestamp, job_id, partition, job_name, user, wait_time, requested_nodes,
                requested_cpus, requested_mem, reason, is_hardware_limited, tres_per_job
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    job["job_id"],
                    job["partition"],
                    job["job_name"],
                    job["user"],
                    job["wait_time"],
                    job["requested_nodes"],
                    job["requested_cpus"],
                    job["requested_mem"],
                    job["reason"],
                    job["is_hardware_limited"],
                    job["tres_per_job"],
                ),
            )

        # Store cluster stats
        if cluster_stats:
            cursor.execute(
                """
            INSERT INTO cluster_stats (
                timestamp, total_nodes, available_nodes, total_cpus, allocated_cpus,
                idle_cpus, down_cpus, total_memory, allocated_memory, partition_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    cluster_stats["total_nodes"],
                    cluster_stats["available_nodes"],
                    cluster_stats["total_cpus"],
                    cluster_stats["allocated_cpus"],
                    cluster_stats["idle_cpus"],
                    cluster_stats["down_cpus"],
                    cluster_stats["total_memory"],
                    cluster_stats["allocated_memory"],
                    cluster_stats["partition_data"],
                ),
            )

        conn.commit()
        logger.info(
            f"Data stored successfully: {len(pending_jobs)} jobs and cluster stats"
        )
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing data: {e}")
    finally:
        conn.close()


def main():
    """Main function to collect and store data at regular intervals"""
    parser = argparse.ArgumentParser(description="Collect Slurm wait time data")
    parser.add_argument(
        "--db", default="slurm_wait_times.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--interval", type=int, default=300, help="Collection interval in seconds"
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=0,
        help="Maximum runtime in seconds (0 = run forever)",
    )
    args = parser.parse_args()

    logger.info("Starting Slurm wait time data collector")
    logger.info(f"Database: {args.db}")
    logger.info(f"Collection interval: {args.interval} seconds")

    # Create database if it doesn't exist
    create_database(args.db)

    start_time = time.time()
    iteration = 0

    try:
        while True:
            iteration += 1
            current_time = int(time.time())

            logger.info(f"Starting collection iteration {iteration}")

            # Collect data
            pending_jobs = get_pending_jobs()
            cluster_stats = get_cluster_stats()

            # Store data
            store_data(args.db, current_time, pending_jobs, cluster_stats)

            logger.info(f"Completed collection iteration {iteration}")

            # Check if we should exit
            if args.max_runtime > 0 and (time.time() - start_time) >= args.max_runtime:
                logger.info("Maximum runtime reached, exiting")
                break

            # Sleep until next interval
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Data collection stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
