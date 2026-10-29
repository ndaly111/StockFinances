# One-time: register the Windows task that runs the Top Financial News brief.
# Run from a normal PowerShell window (no admin needed):
#   powershell -ExecutionPolicy Bypass -File scripts\register_market_brief_task.ps1
# Mirrors the EarningsTakeaways task: interactive logon, start-when-available,
# runs daily at 07:00 and 12:00 ET (Nick, 2026-10-28).
$a = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Users\ndaly\projects\sf-earnings-bot\scripts\run_market_brief.ps1"'
$t1 = New-ScheduledTaskTrigger -Daily -At '07:00'
$t2 = New-ScheduledTaskTrigger -Daily -At '12:00'
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'MarketBrief' -Action $a -Trigger $t1,$t2 -Settings $s `
    -Description 'Top Financial News brief for nicksstockfinancials.com (market_brief.py, claude -p on subscription)' -Force | Out-Null
$t = Get-ScheduledTask -TaskName 'MarketBrief'
"MarketBrief registered: state=$($t.State)"
$t.Triggers | ForEach-Object { "  trigger: daily at $($_.StartBoundary.Substring(11,5))" }
"  next run: $((Get-ScheduledTaskInfo -TaskName 'MarketBrief').NextRunTime)"
