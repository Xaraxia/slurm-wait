#!/usr/bin/env python3
"""
Slurm Wait Time Report Generator
--------------------------------
Generates management-ready reports and visualizations from collected Slurm wait time data.

module load anaconda3 to make it work

# Generate a report with all available data
python slurm-wait-report.py

# Generate a report for a specific date range
python slurm-wait-report.py --start-date 2025-03-01 --end-date 2025-04-01

# Specify output directory and database location
python slurm-wait-report.py --db /path/to/database.db --output-dir /path/to/reports
"""

import sqlite3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("slurm_report.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("slurm_report")

# Set the style for prettier plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

def format_time(seconds, pos=None):
    """Format seconds as hours:minutes for plot labels"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    return f"{hours}h {minutes}m"

def load_data(db_path, start_date=None, end_date=None):
    """Load data from the database for the specified date range"""
    conn = sqlite3.connect(db_path)

    # Define date filters
    date_filter = ""
    params = []

    if start_date and end_date:
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86400  # Include the full end date
        date_filter = "WHERE timestamp >= ? AND timestamp < ?"
        params = [start_timestamp, end_timestamp]

    # Load pending jobs - using subquery to get only the latest record for each job_id
    jobs_query = f"""
    SELECT
        j.timestamp,
        j.job_id,
        j.partition,
        j.user,
        j.wait_time,
        j.requested_cpus,
        j.requested_mem,
        j.reason,
        j.is_hardware_limited
    FROM pending_jobs j
    INNER JOIN (
        SELECT job_id, MAX(timestamp) as max_timestamp
        FROM pending_jobs
        {date_filter}
        GROUP BY job_id
    ) latest ON j.job_id = latest.job_id AND j.timestamp = latest.max_timestamp
    ORDER BY j.timestamp
    """

    jobs_df = pd.read_sql_query(jobs_query, conn, params=params)

    # Convert timestamp to datetime
    jobs_df['datetime'] = pd.to_datetime(jobs_df['timestamp'], unit='s')
    jobs_df['date'] = jobs_df['datetime'].dt.date

    # Load cluster stats
    stats_query = f"""
    SELECT
        timestamp,
        total_nodes,
        available_nodes,
        total_cpus,
        allocated_cpus,
        idle_cpus,
        down_cpus,
        total_memory,
        allocated_memory,
        partition_data
    FROM cluster_stats
    {date_filter}
    ORDER BY timestamp
    """

    stats_df = pd.read_sql_query(stats_query, conn, params=params)

    # Convert timestamp to datetime
    stats_df['datetime'] = pd.to_datetime(stats_df['timestamp'], unit='s')
    stats_df['date'] = stats_df['datetime'].dt.date

    conn.close()

    return jobs_df, stats_df

def plot_average_wait_times(jobs_df, output_dir):
    """Plot average wait times for hardware-limited jobs over time"""
    # Filter for hardware-limited jobs only
    hardware_jobs = jobs_df[jobs_df['is_hardware_limited'] == 1]
    
    if hardware_jobs.empty:
        logger.warning("No hardware-limited jobs found for the time period")
        return
    
    # Resample data by hour and calculate average wait time
    hourly_wait_times = hardware_jobs.set_index('datetime').resample('1H').agg({'wait_time': 'median'}).reset_index()
    hourly_wait_times['wait_time_hours'] = hourly_wait_times['wait_time'] / 3600
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_wait_times['datetime'], hourly_wait_times['wait_time'], linewidth=2)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_time))
    plt.title('Median Wait Time for Hardware-Limited Jobs', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Wait Time (hours:minutes)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'average_wait_times.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Median wait times plot saved to {plot_path}")

def plot_wait_time_distribution(jobs_df, output_dir):
    """Plot distribution of wait times for hardware-limited jobs"""
    # Filter for hardware-limited jobs only
    hardware_jobs = jobs_df[jobs_df['is_hardware_limited'] == 1]
    
    if hardware_jobs.empty:
        logger.warning("No hardware-limited jobs found for the time period")
        return
    
    # Convert wait time to hours for better readability
    wait_hours = hardware_jobs['wait_time'] / 3600
    
    plt.figure(figsize=(10, 6))
    sns.histplot(wait_hours, bins=30, kde=True)
    plt.title('Distribution of Wait Times for Hardware-Limited Jobs', fontsize=14, fontweight='bold')
    plt.xlabel('Wait Time (hours)', fontsize=12)
    plt.ylabel('Count of Jobs', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'wait_time_distribution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Wait time distribution plot saved to {plot_path}")

def plot_wait_time_by_partition(jobs_df, output_dir):
    """Plot average wait times by partition"""
    # Filter for hardware-limited jobs only
    hardware_jobs = jobs_df[jobs_df['is_hardware_limited'] == 1]
    
    if hardware_jobs.empty:
        logger.warning("No hardware-limited jobs found for the time period")
        return
    
    # Group by partition and calculate stats
    partition_stats = hardware_jobs.groupby('partition').agg(
        avg_wait=('wait_time', 'median'),
        max_wait=('wait_time', 'max'),
        job_count=('job_id', 'count')
    ).reset_index()
    
    # Sort by average wait time
    partition_stats = partition_stats.sort_values('avg_wait', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create positions for bars
    positions = np.arange(len(partition_stats))
    
    # Plot average wait time bars
    avg_bars = plt.bar(positions, partition_stats['avg_wait'], width=0.6, alpha=0.7, color='steelblue')
    
    # Add job count as text
    for i, (avg, count) in enumerate(zip(partition_stats['avg_wait'], partition_stats['job_count'])):
        plt.text(i, avg + avg * 0.05, f"{count} jobs", ha='center', va='bottom', fontsize=9)
    
    # Configure axes
    plt.title('Median Wait Time by Partition (Hardware Limited Jobs Only)', fontsize=14, fontweight='bold')
    plt.xlabel('Partition', fontsize=12)
    plt.ylabel('Median Wait Time (seconds)', fontsize=12)
    plt.xticks(positions, partition_stats['partition'], rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_time))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'wait_time_by_partition.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Wait time by partition plot saved to {plot_path}")

def plot_resource_utilization(stats_df, output_dir):
    """Plot cluster resource utilization over time"""
    if stats_df.empty:
        logger.warning("No cluster stats found for the time period")
        return
    
    # Calculate CPU utilization
    stats_df['cpu_utilization'] = stats_df['allocated_cpus'] / stats_df['total_cpus'] * 100
    stats_df['memory_utilization'] = stats_df['allocated_memory'] / stats_df['total_memory'] * 100
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # CPU utilization
    ax1.plot(stats_df['datetime'], stats_df['cpu_utilization'], linewidth=2, color='#1f77b4')
    ax1.set_title('CPU Utilization Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('CPU Utilization (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Memory utilization
    ax2.plot(stats_df['datetime'], stats_df['memory_utilization'], linewidth=2, color='#ff7f0e')
    ax2.set_title('Memory Utilization Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Memory Utilization (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'resource_utilization.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Resource utilization plot saved to {plot_path}")

def plot_correlation_matrix(jobs_df, stats_df, output_dir):
    """Plot correlation between resource utilization and wait times"""
    # Only use hardware-limited jobs
    hardware_jobs = jobs_df[jobs_df['is_hardware_limited'] == 1]
    
    if hardware_jobs.empty or stats_df.empty:
        logger.warning("Not enough data to generate correlation matrix")
        return
    
    # Group jobs by hour
    hourly_jobs = hardware_jobs.set_index('datetime').resample('1H').agg({
        'wait_time': 'median',
        'job_id': 'count'
    }).reset_index()
    hourly_jobs.rename(columns={'job_id': 'job_count'}, inplace=True)
    
    # Group stats by hour
    hourly_stats = stats_df.set_index('datetime').resample('1H').agg({
        'cpu_utilization': 'median',
        'memory_utilization': 'median',
        'total_cpus': 'median',
        'allocated_cpus': 'median',
        'idle_cpus': 'median',
        'down_cpus': 'median'
    }).reset_index()
    
    # Merge the datasets
    merged_df = pd.merge_asof(
        hourly_jobs.sort_values('datetime'),
        hourly_stats.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )
    
    # Calculate correlation matrix
    corr_columns = ['wait_time', 'job_count', 'cpu_utilization', 'memory_utilization', 
                   'total_cpus', 'allocated_cpus', 'idle_cpus', 'down_cpus']
    correlation = merged_df[corr_columns].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Cluster Metrics and Wait Times', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Correlation matrix saved to {plot_path}")

def plot_wait_time_vs_utilization(jobs_df, stats_df, output_dir):
    """Plot wait time vs. CPU utilization scatter plot"""
    # Only use hardware-limited jobs
    hardware_jobs = jobs_df[jobs_df['is_hardware_limited'] == 1]
    
    if hardware_jobs.empty or stats_df.empty:
        logger.warning("Not enough data to generate wait time vs. utilization plot")
        return
    
    # Group jobs by hour
    hourly_jobs = hardware_jobs.set_index('datetime').resample('1H').agg({
        'wait_time': 'median'
    }).reset_index()
    
    # Get CPU utilization by hour
    stats_df['cpu_utilization'] = stats_df['allocated_cpus'] / stats_df['total_cpus'] * 100
    hourly_stats = stats_df.set_index('datetime').resample('1H').agg({
        'cpu_utilization': 'median'
    }).reset_index()
    
    # Merge the datasets
    merged_df = pd.merge_asof(
        hourly_jobs.sort_values('datetime'),
        hourly_stats.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )
    
    # Drop any NaN values that might cause issues
    merged_df = merged_df.dropna(subset=['cpu_utilization', 'wait_time'])
    
    # Check if we have enough data
    if len(merged_df) < 2:
        logger.warning("Not enough valid data points to plot wait time vs. CPU utilization")
        plt.figure(figsize=(10, 8))
        plt.title('Median Wait Time vs. CPU Utilization\n(Insufficient data for trend analysis)', fontsize=14, fontweight='bold')
        plt.xlabel('CPU Utilization (%)', fontsize=12)
        plt.ylabel('Median Wait Time (hours)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'wait_time_vs_utilization.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logger.info(f"Wait time vs. utilization plot saved to {plot_path}")
        return
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(merged_df['cpu_utilization'], merged_df['wait_time'] / 3600, alpha=0.6)
    
    # Add trend line - use try/except to handle potential numerical errors
    try:
        # Check for perfect collinearity
        if merged_df['cpu_utilization'].std() > 0.001:  # Ensure some variation in x values
            z = np.polyfit(merged_df['cpu_utilization'], merged_df['wait_time'] / 3600, 1)
            p = np.poly1d(z)
            x_range = np.linspace(merged_df['cpu_utilization'].min(), merged_df['cpu_utilization'].max(), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.8)
            
            # Add regression equation to the plot
            equation = f"y = {z[0]:.4f}x + {z[1]:.4f}"
            plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        else:
            logger.warning("Not enough variation in CPU utilization for trend line")
    except Exception as e:
        logger.warning(f"Could not generate trend line: {e}")
    
    plt.title('Median Wait Time vs. CPU Utilization', fontsize=14, fontweight='bold')
    plt.xlabel('CPU Utilization (%)', fontsize=12)
    plt.ylabel('Median Wait Time (hours)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'wait_time_vs_utilization.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Wait time vs. utilization plot saved to {plot_path}")

def generate_summary_stats(jobs_df, stats_df):
    """Generate summary statistics for the report"""
    summary = {}
    
    # Filter for hardware-limited jobs only
    hardware_jobs = jobs_df[jobs_df['is_hardware_limited'] == 1]
    
    if not hardware_jobs.empty:
        # Overall wait time stats
        summary['total_jobs'] = len(hardware_jobs)
        summary['avg_wait_time'] = hardware_jobs['wait_time'].median()
        summary['median_wait_time'] = hardware_jobs['wait_time'].median()
        summary['max_wait_time'] = hardware_jobs['wait_time'].max()
        
        # Calculate 90th percentile wait time
        summary['p90_wait_time'] = hardware_jobs['wait_time'].quantile(0.9)

# Wait time by user
        user_stats = hardware_jobs.groupby('user').agg(
            avg_wait=('wait_time', 'median'),
            job_count=('job_id', 'count')
        ).sort_values('avg_wait', ascending=False)
        summary['top_waiting_users'] = user_stats.head(5).to_records()

        # Wait time by partition
        partition_stats = hardware_jobs.groupby('partition').agg(
            avg_wait=('wait_time', 'median'),
            job_count=('job_id', 'count')
        ).sort_values('avg_wait', ascending=False)
        summary['partition_wait_times'] = partition_stats.to_records()
    else:
        summary['total_jobs'] = 0
        summary['avg_wait_time'] = 0
        summary['median_wait_time'] = 0
        summary['max_wait_time'] = 0
        summary['p90_wait_time'] = 0
        summary['top_waiting_users'] = []
        summary['partition_wait_times'] = []
    
    # Resource utilization stats
    if not stats_df.empty:
        # Calculate utilization metrics
        stats_df['cpu_utilization'] = stats_df['allocated_cpus'] / stats_df['total_cpus'] * 100
        stats_df['memory_utilization'] = stats_df['allocated_memory'] / stats_df['total_memory'] * 100
        
        summary['avg_cpu_utilization'] = stats_df['cpu_utilization'].median()
        summary['max_cpu_utilization'] = stats_df['cpu_utilization'].max()
        summary['avg_memory_utilization'] = stats_df['memory_utilization'].median()
        summary['max_memory_utilization'] = stats_df['memory_utilization'].max()
        
        # Calculate percentage of time above 90% utilization
        high_util_time = (stats_df['cpu_utilization'] > 90).sum() / len(stats_df) * 100
        summary['high_util_percentage'] = high_util_time
    else:
        summary['avg_cpu_utilization'] = 0
        summary['max_cpu_utilization'] = 0
        summary['avg_memory_utilization'] = 0
        summary['max_memory_utilization'] = 0
        summary['high_util_percentage'] = 0
    
    return summary

def generate_html_report(jobs_df, stats_df, summary, output_dir, start_date, end_date):
    """Generate an HTML report with plots and statistics"""
    # Format dates for display
    display_start = start_date if start_date else "earliest data"
    display_end = end_date if end_date else "latest data"
    
    # Format summary statistics for display
    avg_wait_hours = summary['avg_wait_time'] / 3600
    median_wait_hours = summary['median_wait_time'] / 3600
    max_wait_hours = summary['max_wait_time'] / 3600
    p90_wait_hours = summary['p90_wait_time'] / 3600
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HPC Cluster Wait Time Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #ddd;
            }}
            .summary-box {{
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 30px;
            }}
            .stat-row {{
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 15px;
            }}
            .stat-item {{
                flex: 1;
                min-width: 250px;
                margin-bottom: 15px;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }}
            .stat-label {{
                font-size: 14px;
                color: #666;
            }}
            .plot-container {{
                margin-bottom: 40px;
            }}
            .plot-img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
                border: 1px solid #ddd;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .conclusion {{
                background-color: #f0f7fb;
                border-left: 5px solid #3498db;
                padding: 15px;
                margin-bottom: 30px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>HPC Cluster Wait Time Analysis Report</h1>
            <p>Report Period: {display_start} to {display_end}</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>This report analyzes job wait times due to insufficient hardware resources rather than policy limitations.</p>
            
            <div class="stat-row">
                <div class="stat-item">
                    <div class="stat-value">{summary['total_jobs']}</div>
                    <div class="stat-label">Hardware-Limited Jobs</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{avg_wait_hours:.2f} hours</div>
                    <div class="stat-label">Median Wait Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{median_wait_hours:.2f} hours</div>
                    <div class="stat-label">Median Wait Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{p90_wait_hours:.2f} hours</div>
                    <div class="stat-label">90th Percentile Wait Time</div>
                </div>
            </div>
            
            <div class="stat-row">
                <div class="stat-item">
                    <div class="stat-value">{summary['avg_cpu_utilization']:.1f}%</div>
                    <div class="stat-label">Median CPU Utilization</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{summary['max_cpu_utilization']:.1f}%</div>
                    <div class="stat-label">Peak CPU Utilization</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{summary['high_util_percentage']:.1f}%</div>
                    <div class="stat-label">Time at >90% Utilization</div>
                </div>
            </div>
        </div>
        
        <h2>Wait Time Analysis</h2>
        
        <div class="plot-container">
            <h3>Median Wait Time Trend</h3>
            <img class="plot-img" src="average_wait_times.png" alt="Median Wait Times">
            <p>This chart shows the trend of average wait times for hardware-limited jobs over the reporting period.</p>
        </div>
        
        <div class="plot-container">
            <h3>Wait Time Distribution</h3>
            <img class="plot-img" src="wait_time_distribution.png" alt="Wait Time Distribution">
            <p>Distribution of wait times across all hardware-limited jobs, showing the frequency of different wait durations.</p>
        </div>
        
        <div class="plot-container">
            <h3>Wait Time by Partition</h3>
            <img class="plot-img" src="wait_time_by_partition.png" alt="Wait Time by Partition">
            <p>Comparison of average wait times across different partitions, highlighting which partitions experience the longest delays.</p>
        </div>
        
        <h2>Resource Utilization</h2>
        
        <div class="plot-container">
            <h3>CPU and Memory Utilization</h3>
            <img class="plot-img" src="resource_utilization.png" alt="Resource Utilization">
            <p>CPU and memory utilization over time, showing when the cluster is at or near capacity.</p>
        </div>
        
        <div class="plot-container">
            <h3>Wait Time vs. CPU Utilization</h3>
            <img class="plot-img" src="wait_time_vs_utilization.png" alt="Wait Time vs Utilization">
            <p>Scatter plot showing the relationship between CPU utilization and job wait times.</p>
        </div>
        
        <div class="plot-container">
            <h3>Correlation Analysis</h3>
            <img class="plot-img" src="correlation_matrix.png" alt="Correlation Matrix">
            <p>Correlation matrix showing relationships between various metrics, including wait times and resource utilization.</p>
        </div>
        
        <h2>Detailed Statistics</h2>
        
        <h3>Top Users Experiencing Wait Times</h3>
        <table>
            <tr>
                <th>User</th>
                <th>Median Wait Time (hours)</th>
                <th>Number of Jobs</th>
            </tr>
    """
    
    # Add top users data
    for user_data in summary['top_waiting_users']:
        wait_hours = user_data['avg_wait'] / 3600
        userd = user_data['user']
        userjc = user_data['job_count']
        html_content += f"""
            <tr>
                <td>{userd}</td>
                <td>{wait_hours:.2f}</td>
                <td>{userjc}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>Wait Times by Partition</h3>
        <table>
            <tr>
                <th>Partition</th>
                <th>Median Wait Time (hours)</th>
                <th>Number of Jobs</th>
            </tr>
    """
    
    # Add partition data
    for part_data in summary['partition_wait_times']:
        wait_hours = part_data['avg_wait'] / 3600
        html_content += f"""
            <tr>
                <td>{part_data['partition']}</td>
                <td>{wait_hours:.2f}</td>
                <td>{part_data['job_count']}</td>
            </tr>
        """
    
    # Add conclusion and recommendations based on data
    high_util_threshold = 75
    long_wait_threshold = 4  # 4 hours
    
    recommendations = []
    if summary['avg_cpu_utilization'] > high_util_threshold:
        recommendations.append("Consider expanding cluster capacity as average CPU utilization exceeds 75%.")
    
    if summary['high_util_percentage'] > 30:
        recommendations.append(f"The cluster is operating above 90% capacity for {summary['high_util_percentage']:.1f}% of the time, suggesting hardware resources are consistently stretched.")
    
    if avg_wait_hours > long_wait_threshold:
        recommendations.append(f"Median wait times are high at {avg_wait_hours:.2f} hours, indicating insufficient resources to meet demand.")
    
    # Add partition-specific recommendations
    partition_recommendations = []
    for part_data in summary['partition_wait_times']:
        wait_hours = part_data['avg_wait'] / 3600
        if wait_hours > long_wait_threshold and part_data['job_count'] > 10:
            partition_recommendations.append(f"Partition '{part_data['partition']}' has particularly high wait times ({wait_hours:.2f} hours) across {part_data['job_count']} jobs.")
    
    html_content += """
        </table>
        
        <div class="conclusion">
            <h2>Conclusions and Recommendations</h2>
    """
    
    if recommendations:
        html_content += "<h3>Overall Recommendations:</h3><ul>"
        for rec in recommendations:
            html_content += f"<li>{rec}</li>"
        html_content += "</ul>"
    
    if partition_recommendations:
        html_content += "<h3>Partition-Specific Recommendations:</h3><ul>"
        for rec in partition_recommendations:
            html_content += f"<li>{rec}</li>"
        html_content += "</ul>"
    
    if not recommendations and not partition_recommendations:
        html_content += "<p>Based on the current data, the cluster appears to be sufficiently sized for the workload. Wait times are within acceptable limits and resource utilization is balanced.</p>"
    
    html_content += """
        </div>
        
        <div class="footer">
            <p>Generated by Slurm Wait Time Report Generator | For internal use only</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report to file
    report_path = os.path.join(output_dir, 'slurm_wait_time_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {report_path}")
    
    return report_path

def main():
    """Main function to generate reports and visualizations"""
    parser = argparse.ArgumentParser(description='Generate Slurm wait time reports')
    parser.add_argument('--db', default='slurm_wait_times.db', help='Path to SQLite database')
    parser.add_argument('--start-date', help='Start date for report (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for report (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='report', help='Output directory for reports')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logger.info(f"Starting report generation")
    logger.info(f"Database: {args.db}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load data from database
        jobs_df, stats_df = load_data(args.db, args.start_date, args.end_date)
        
        logger.info(f"Loaded {len(jobs_df)} jobs and {len(stats_df)} cluster stat entries")
        
        # Generate plots
        plot_average_wait_times(jobs_df, args.output_dir)
        plot_wait_time_distribution(jobs_df, args.output_dir)
        plot_wait_time_by_partition(jobs_df, args.output_dir)
        plot_resource_utilization(stats_df, args.output_dir)
        plot_correlation_matrix(jobs_df, stats_df, args.output_dir)
        plot_wait_time_vs_utilization(jobs_df, stats_df, args.output_dir)
        
        # Generate summary statistics
        summary = generate_summary_stats(jobs_df, stats_df)
        
        # Generate HTML report
        report_path = generate_html_report(
            jobs_df, stats_df, summary, args.output_dir, args.start_date, args.end_date
        )
        
        logger.info(f"Report generation completed successfully")
        logger.info(f"Report is available at: {report_path}")
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise

if __name__ == "__main__":
    main()
