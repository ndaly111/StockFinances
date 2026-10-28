# Scheduled runner for earnings_takeaways.py (Windows task "EarningsTakeaways").
# Runs in a dedicated worktree on main; the writer uses `claude -p` on Nick's
# subscription. Pass -Backfill N on the overnight run to fill tickers with no card.
param([int]$Backfill = 0)
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\ndaly\projects\sf-earnings-bot'
$log = 'C:\Users\ndaly\projects\sf-earnings-bot-logs\takeaways.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm') backfill=$Backfill" | Add-Content $log
git pull -q --rebase origin main *>> $log
$env:PYTHONIOENCODING = 'utf-8'
$args = @('earnings_takeaways.py', '--push')
if ($Backfill -gt 0) { $args += @('--backfill', "$Backfill") }
& py -3.12 @args *>> $log
