param(
    [ValidateSet('Apply', 'Restore')][string]$Action = 'Apply',
    [string]$ReceiptDirectory = 'E:\capsule\runs\system\paging-f-20261009'
)

$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator approval is required.'
}
$memoryKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$desiredPaging = @('C:\pagefile.sys 2048 2048', 'F:\pagefile.sys 16384 32768')
$backupPath = Join-Path $ReceiptDirectory 'before.json'
$resultPath = Join-Path $ReceiptDirectory ($Action.ToLowerInvariant() + '-result.json')

function Write-Receipt($Path, $Value) {
    $text = $Value | ConvertTo-Json -Depth 8
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $writer = New-Object IO.StreamWriter($stream, (New-Object Text.UTF8Encoding($false)))
        try { $writer.WriteLine($text) } finally { $writer.Dispose() }
    } finally { $stream.Dispose() }
}

function Read-State {
    $computer = Get-CimInstance Win32_ComputerSystem
    $registry = Get-ItemProperty -LiteralPath $memoryKey -Name PagingFiles
    return @{
        observed_at = [DateTime]::UtcNow.ToString('o')
        automatic_managed_pagefile = [bool]$computer.AutomaticManagedPagefile
        paging_files = @($registry.PagingFiles)
        configured = @(Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize)
        active = @(Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage)
        crash_dump_enabled = (Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' -Name CrashDumpEnabled).CrashDumpEnabled
    }
}

function Set-State($Automatic, $Entries) {
    Get-CimInstance Win32_ComputerSystem | Set-CimInstance -Property @{AutomaticManagedPagefile = [bool]$Automatic}
    Set-ItemProperty -LiteralPath $memoryKey -Name PagingFiles -Value ([string[]]$Entries)
}

if (Test-Path -LiteralPath $resultPath) { throw 'This operation already has a receipt; refusing overwrite.' }
if ($Action -eq 'Apply') {
    if (Test-Path -LiteralPath $ReceiptDirectory) { throw 'Receipt directory already exists; refusing a repeated apply.' }
    $volume = Get-Volume -DriveLetter F
    if ($volume.FileSystem -ne 'NTFS' -or $volume.DriveType -ne 'Fixed' -or $volume.HealthStatus -ne 'Healthy' -or $volume.SizeRemaining -lt 40GB) {
        throw 'F: is not a healthy fixed NTFS volume with at least 40 GiB free.'
    }
    $before = Read-State
    if (-not $before.automatic_managed_pagefile -or $before.crash_dump_enabled -ne 3) {
        throw 'Current configuration differs from the inspected automatic paging / small-dump configuration.'
    }
    if (@($before.active | Where-Object { $_.Name -ne 'C:\pagefile.sys' }).Count -ne 0) {
        throw 'Unexpected additional active pagefile; refusing to replace its settings.'
    }
    New-Item -ItemType Directory -Path $ReceiptDirectory | Out-Null
    Write-Receipt $backupPath $before
    try {
        Set-State $false $desiredPaging
        $after = Read-State
        if ($after.automatic_managed_pagefile -or @(Compare-Object $desiredPaging $after.paging_files).Count -ne 0 -or $after.crash_dump_enabled -ne $before.crash_dump_enabled) {
            throw 'Readback did not match the requested configuration.'
        }
        Write-Receipt $resultPath @{
            success = $true; action = 'Apply'; restart_required = $true
            active_files_deleted = $false; system_restarted = $false
            before_path = $backupPath; after = $after
        }
    } catch {
        $failure = $_.Exception.Message
        $restored = $false
        try {
            Set-State $before.automatic_managed_pagefile $before.paging_files
            $rollback = Read-State
            $restored = ($rollback.automatic_managed_pagefile -eq $before.automatic_managed_pagefile -and @(Compare-Object $before.paging_files $rollback.paging_files).Count -eq 0)
        } catch { $failure += '; rollback error: ' + $_.Exception.Message }
        Write-Receipt $resultPath @{success = $false; action = 'Apply'; error = $failure; original_configuration_restored = $restored}
        throw $failure
    }
} else {
    if (-not (Test-Path -LiteralPath $backupPath)) { throw 'Original configuration backup is missing.' }
    $before = Get-Content -LiteralPath $backupPath -Raw | ConvertFrom-Json
    $current = Read-State
    if ($current.automatic_managed_pagefile -or @(Compare-Object $desiredPaging $current.paging_files).Count -ne 0) {
        throw 'Paging settings changed since this operation; refusing to overwrite another change.'
    }
    Set-State $before.automatic_managed_pagefile $before.paging_files
    $after = Read-State
    if ($after.automatic_managed_pagefile -ne $before.automatic_managed_pagefile -or @(Compare-Object $before.paging_files $after.paging_files).Count -ne 0) {
        throw 'Restoration readback mismatch.'
    }
    Write-Receipt $resultPath @{success = $true; action = 'Restore'; restart_required = $true; after = $after}
}
Write-Output 'Paging configuration saved. Restart Windows manually to activate it.'
