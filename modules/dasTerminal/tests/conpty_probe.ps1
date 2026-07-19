$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::Write("DASHERD_READY:$PID`r`n")
$line = [Console]::ReadLine()
[Console]::Write("DASHERD_ECHO:$line`r`n")
exit 23
