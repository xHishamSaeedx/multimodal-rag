# PowerShell script to ingest documents into Neo4j knowledge graph
# Usage: .\ingest.ps1 "path\to\document.pdf"

param(
    [Parameter(Mandatory=$true)]
    [string]$FilePath
)

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Run the ingestion script
python ingest_document.py $FilePath

