Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$screenshot = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
$graphics = [System.Drawing.Graphics]::FromImage($screenshot)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$screenshot.Save("C:\Users\skyeAM\SkyeAM Dropbox\SAMresearch\ABtoPython\tradingCode\final_dashboard.png")
$graphics.Dispose()
$screenshot.Dispose()
Write-Host "Screenshot saved to final_dashboard.png"