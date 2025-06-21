#!/usr/bin/env python3
"""
Example usage of the PDF upload endpoint
"""

import requests
import time
import json
from pathlib import Path


def upload_pdf_and_track_progress(pdf_file_path: str, base_url: str = "http://localhost:8000"):
    """
    Upload a PDF and track processing progress

    Args:
        pdf_file_path: Path to the PDF file to upload
        base_url: Base URL of the API server
    """

    # Step 1: Upload the PDF
    print(f"📤 Uploading PDF: {pdf_file_path}")

    with open(pdf_file_path, 'rb') as f:
        files = {'file': (Path(pdf_file_path).name, f, 'application/pdf')}

        response = requests.post(
            f"{base_url}/api/v1/pdf/upload-pdf",
            files=files
        )

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.json()}")
        return None

    upload_data = response.json()
    job_id = upload_data['job_id']

    print(f"✅ Upload successful!")
    print(f"   Job ID: {job_id}")
    print(f"   Status URL: {upload_data['status_url']}")

    # Step 2: Track processing progress
    print(f"\n🔄 Tracking processing progress...")

    last_stage = None
    while True:
        # Get current status
        status_response = requests.get(f"{base_url}/api/v1/pdf/status/{job_id}")

        if status_response.status_code != 200:
            print(f"❌ Failed to get status: {status_response.json()}")
            break

        status_data = status_response.json()
        stage = status_data['stage']
        progress = status_data['progress']
        message = status_data['message']

        # Print progress if stage changed
        if stage != last_stage:
            print(f"📊 {stage.upper()}: {message}")
            last_stage = stage

        # Show progress bar
        progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
        print(f"\r   [{progress_bar}] {progress * 100:.1f}%", end="", flush=True)

        # Check if completed or failed
        if stage == "completed":
            print(f"\n🎉 Processing completed successfully!")

            files = status_data.get('files', {})
            print(f"📁 Generated files:")
            for file_type, file_path in files.items():
                print(f"   - {file_type.upper()}: {file_path}")

            return job_id, files

        elif stage == "failed":
            print(f"\n❌ Processing failed!")
            print(f"   Error: {status_data.get('error', 'Unknown error')}")
            return None

        # Wait before next check
        time.sleep(1)

    return None


def download_generated_files(job_id: str, base_url: str = "http://localhost:8000"):
    """
    Download information about generated files

    Args:
        job_id: Job ID from upload
        base_url: Base URL of the API server
    """

    file_types = ['mp4', 'manifest', 'faiss']

    print(f"\n📥 Checking available downloads for job {job_id}...")

    for file_type in file_types:
        response = requests.get(f"{base_url}/api/v1/pdf/download/{job_id}/{file_type}")

        if response.status_code == 200:
            file_info = response.json()
            print(f"✅ {file_type.upper()} file available:")
            print(f"   Path: {file_info['file_path']}")
            print(f"   Size: {file_info['file_size']:,} bytes")
        else:
            print(f"⚠️  {file_type.upper()} file not available")


def list_all_jobs(base_url: str = "http://localhost:8000"):
    """
    List all processing jobs

    Args:
        base_url: Base URL of the API server
    """

    response = requests.get(f"{base_url}/api/v1/pdf/list-jobs")

    if response.status_code == 200:
        jobs_data = response.json()
        print(f"<UNK> Jobs:", jobs_data)
        print(f"\n📋 Active jobs: {jobs_data['total']}")
        for job_id in jobs_data['jobs']:
            print(f"   - {job_id}")
    else:
        print(f"❌ Failed to list jobs: {response.json()}")


def main():
    """Main example function"""

    print("🎯 PDF Upload and Processing Example")
    print("=" * 50)

    # Example 1: Upload and process a PDF
    pdf_path = "sample_document.pdf"  # Replace with your PDF path

    if Path(pdf_path).exists():
        result = upload_pdf_and_track_progress(pdf_path)

        if result:
            job_id, files = result

            # Download file information
            download_generated_files(job_id)

            print(f"\n✨ Success! Your PDF has been converted to video format.")
            print(f"   You can now use the generated files for RAG operations.")
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        print(f"   Please create a sample PDF or update the path.")

    # Example 2: List all jobs
    list_all_jobs()


def create_sample_pdf(filename: str = "sample_document.pdf"):
    """
    Create a sample PDF for testing

    Args:
        filename: Output filename for the PDF
    """

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    print(f"📝 Creating sample PDF: {filename}")

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Sample Document for RAG Processing")

    # Add content
    c.setFont("Helvetica", 12)
    y_position = height - 100

    content = [
        "This is a sample document for testing the PDF upload endpoint.",
        "",
        "Machine Learning Concepts:",
        "• Supervised learning uses labeled data to train models",
        "• Unsupervised learning finds patterns in unlabeled data",
        "• Reinforcement learning learns through rewards and penalties",
        "",
        "Deep Learning:",
        "Neural networks with multiple layers can model complex patterns.",
        "Convolutional networks excel at image processing tasks.",
        "Recurrent networks are suited for sequential data analysis.",
        "",
        "Natural Language Processing:",
        "Text preprocessing includes tokenization and normalization.",
        "Word embeddings capture semantic relationships between words.",
        "Transformer models have revolutionized language understanding.",
        "",
        "This document contains enough content to generate multiple",
        "text chunks when processed by the video encoder system."
    ]

    for line in content:
        c.drawString(50, y_position, line)
        y_position -= 20

        if y_position < 50:  # Start new page if needed
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 50

    c.save()
    print(f"✅ Sample PDF created: {filename}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create-sample":
        create_sample_pdf()
    else:
        # Create sample PDF if it doesn't exist
        if not Path("sample_document.pdf").exists():
            create_sample_pdf()

        main()