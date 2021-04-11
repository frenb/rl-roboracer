Start-Process docker-compose stop
$pN = "Streaming"
$allProcesses = get-process -name $pN -errorAction SilentlyContinue
write $allProcesses
foreach ($oneProcess in $allProcesses) {
    write $oneProcess
    $oneProcess.kill()
} 
write "done"
