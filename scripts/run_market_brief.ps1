# Scheduled runner for market_brief.py (Windows task "MarketBrief").
# Runs in the same dedicated worktree on main as the earnings takeaways; the
# writer is `claude -p` on Nick's subscription. Scheduled 07:00 and 12:00 ET;
# the script itself skips the model call if the headlines are unchanged.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\ndaly\projects\sf-earnings-bot'
$log = 'C:\Users\ndaly\projects\sf-earnings-bot-logs\market_brief.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Add-Content $log
git pull -q --rebase origin main *>> $log
$env:PYTHONIOENCODING = 'utf-8'
& py -3.12 market_brief.py --push *>> $log
# keep the log from growing forever (~a few weeks of runs)
if ((Get-Item $log).Length -gt 2MB) { Get-Content $log -Tail 2000 | Set-Content $log }
