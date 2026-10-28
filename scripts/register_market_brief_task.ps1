# One-time: register the Windows task that runs the Top Financial News brief.
# Run from a normal PowerShell window (no admin needed):
#   powershell -ExecutionPolicy Bypass -File scripts\register_market_brief_task.ps1
# Mirrors the EarningsTakeaways task: interactive logon, start-when-available,
# weekdays every 30 min 06:00-18:30 ET, weekends once at 09:00.
$a = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Users\ndaly\projects\sf-earnings-bot\scripts\run_market_brief.ps1"'
$rep = (New-ScheduledTaskTrigger -Once -At '06:00' `
    -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Hours 12.5)).Repetition
$t1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '06:00'
$t1.Repetition = $rep
$t2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday -At '09:00'
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'MarketBrief' -Action $a -Trigger $t1,$t2 -Settings $s `
    -Description 'Top Financial News brief for nicksstockfinancials.com (market_brief.py, claude -p on subscription)' -Force | Out-Null
$t = Get-ScheduledTask -TaskName 'MarketBrief'
"MarketBrief registered: state=$($t.State)"
$t.Triggers | ForEach-Object { "  trigger: days=$($_.DaysOfWeek) start=$($_.StartBoundary.Substring(11,5)) repeat=$($_.Repetition.Interval) for $($_.Repetition.Duration)" }
"  next run: $((Get-ScheduledTaskInfo -TaskName 'MarketBrief').NextRunTime)"
