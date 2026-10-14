param([ValidateSet('Apply', 'Restore')][string]$Action = 'Apply')

$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator approval is required.'
}
$receiptDirectory = 'E:\capsule\runs\system\paging-f-repair-20261011'
$key = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$expected = @('C:\pagefile.sys 2048 2048', 'F:\pagefile.sys 16384 32768')

function Write-Receipt($Name, $Value) {
    $stream = [IO.File]::Open((Join-Path $receiptDirectory $Name), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $writer = New-Object IO.StreamWriter($stream, (New-Object Text.UTF8Encoding($false)))
        try { $writer.WriteLine(($Value | ConvertTo-Json -Depth 8)) } finally { $writer.Dispose() }
    } finally { $stream.Dispose() }
}

function Read-State {
    $registry = Get-ItemProperty -LiteralPath $key
    return @{
        observed_at = [DateTime]::UtcNow.ToString('o')
        pagefile_on_os_volume = $registry.PagefileOnOsVolume
        value_kind = (Get-Item -LiteralPath $key).GetValueKind('PagefileOnOsVolume').ToString()
        paging_files = @($registry.PagingFiles)
        automatic_managed = [bool](Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile
        active = @(Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage)
        crash_dump_enabled = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl').CrashDumpEnabled
    }
}

if (Test-Path -LiteralPath (Join-Path $receiptDirectory ($Action.ToLowerInvariant() + '-result.json'))) {
    throw 'Operation already recorded; refusing overwrite.'
}
$before = Read-State
if ($before.automatic_managed -or @(Compare-Object $expected $before.paging_files).Count -ne 0 -or $before.crash_dump_enabled -ne 3 -or $before.value_kind -ne 'DWord') {
    throw 'Configuration differs from the inspected state; no changes made.'
}
if ($Action -eq 'Apply') {
    if (Test-Path -LiteralPath $receiptDirectory) { throw 'Receipt directory already exists.' }
    if ($before.pagefile_on_os_volume -ne 1) { throw 'Expected override value 1.' }
    $volume = Get-Volume -DriveLetter F
    if ($volume.FileSystem -ne 'NTFS' -or $volume.DriveType -ne 'Fixed' -or $volume.HealthStatus -ne 'Healthy' -or $volume.SizeRemaining -lt 40GB) {
        throw 'F: failed disk protection checks.'
    }
    $encryption = Get-BitLockerVolume -MountPoint 'F:' | Select-Object MountPoint, VolumeStatus, LockStatus, ProtectionStatus, AutoUnlockEnabled
    if ($encryption.LockStatus.ToString() -ne 'Unlocked') { throw 'F: is locked; no changes made.' }
    New-Item -ItemType Directory -Path $receiptDirectory | Out-Null
    Write-Receipt 'before.json' @{state = $before; f_volume = ($volume | Select-Object DriveLetter,FileSystem,HealthStatus,SizeRemaining); encryption = $encryption}
    try {
        Set-ItemProperty -LiteralPath $key -Name PagefileOnOsVolume -Value 0
        $after = Read-State
        if ($after.pagefile_on_os_volume -ne 0 -or $after.value_kind -ne 'DWord' -or $after.automatic_managed -ne $before.automatic_managed -or @(Compare-Object $before.paging_files $after.paging_files).Count -ne 0 -or $after.crash_dump_enabled -ne $before.crash_dump_enabled) {
            throw 'Readback mismatch.'
        }
        Write-Receipt 'apply-result.json' @{configuration_saved = $true; activation_verified = $false; restart_required = $true; system_restarted = $false; changed_value = 'PagefileOnOsVolume: 1 -> 0'; after = $after}
    } catch {
        $failure = $_.Exception.Message
        $restored = $false
        try {
            Set-ItemProperty -LiteralPath $key -Name PagefileOnOsVolume -Value $before.pagefile_on_os_volume
            $restored = (Read-State).pagefile_on_os_volume -eq $before.pagefile_on_os_volume
        } catch { $failure += '; rollback failed: ' + $_.Exception.Message }
        Write-Receipt 'apply-result.json' @{configuration_saved = $false; error = $failure; rollback_verified = $restored}
        throw $failure
    }
} else {
    $backup = Get-Content -LiteralPath (Join-Path $receiptDirectory 'before.json') -Raw | ConvertFrom-Json
    if ($before.pagefile_on_os_volume -ne 0 -or $backup.state.pagefile_on_os_volume -ne 1) { throw 'Unexpected current or backup value.' }
    Set-ItemProperty -LiteralPath $key -Name PagefileOnOsVolume -Value 1
    $after = Read-State
    if ($after.pagefile_on_os_volume -ne 1) { throw 'Restore readback mismatch.' }
    Write-Receipt 'restore-result.json' @{configuration_restored = $true; restart_required = $true; after = $after}
}
Write-Output 'Configuration saved; manually restart Windows, then verify Win32_PageFileUsage includes F:\pagefile.sys. No restart was performed.'
