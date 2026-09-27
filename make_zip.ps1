# Build the final submission zip. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File .\make_zip.ps1 -Team "YourTeamName" -Matching output\matching_results_sib.tsv -Candidates output\candidate_pairs_sib.tsv
param(
  [Parameter(Mandatory=$true)][string]$Team,
  [Parameter(Mandatory=$true)][string]$Matching,
  [Parameter(Mandatory=$true)][string]$Candidates
)
$ErrorActionPreference = "Stop"
$stage = Join-Path $env:TEMP "submission_stage"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force "$stage\output", "$stage\code\business_entity_resolution" | Out-Null

Copy-Item $Matching   "$stage\output\matching_results.tsv"
Copy-Item $Candidates "$stage\output\candidate_pairs.tsv"

$code = ".\code\business_entity_resolution"
Copy-Item "$code\src" "$stage\code\business_entity_resolution\src" -Recurse
Get-ChildItem "$stage\code" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Copy-Item "$code\README.md"        "$stage\code\business_entity_resolution\README.md"
Copy-Item "$code\requirements.txt" "$stage\code\business_entity_resolution\requirements.txt"
Copy-Item ".\Documentation_template.md" "$stage\Documentation_template.md"

$zip = ".\${Team}_submission.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal
"Created $zip  ($([math]::Round((Get-Item $zip).Length/1MB)) MB)"
Get-ChildItem $stage -Recurse -File | ForEach-Object { $_.FullName.Replace("$stage\", "") }
