import platform

def checkArchitecture()-> str:
    machine = platform.machine()
    if machine == 'arm64':
        return 'ARM64'
    if machine == 'i386':
        return 'i386'
    if machine == 'x86_64':
        return 'x86_64'
    return "Architecture not found"