#!/usr/bin/env python3
"""
Smart Package Installer
----------------------
This script reads requirements from requirements.txt, attempts to install them,
and uses Google Cloud Vertex AI to resolve any conflicts or errors encountered.
"""

import os
import sys
import re
import subprocess
import platform
import json
import time
import argparse
import logging
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("installation_log.txt")
    ]
)
logger = logging.getLogger(__name__)

# Check for required Google Cloud libraries
try:
    from google.cloud import aiplatform
    from vertexai.preview.generative_models import GenerativeModel, Part
except ImportError:
    logger.info("Installing Google Cloud Vertex AI SDK...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-aiplatform"])
    # Try import again after installation
    from google.cloud import aiplatform
    from vertexai.preview.generative_models import GenerativeModel, Part

class SmartInstaller:
    def __init__(self, requirements_file: str, project_id: str = None, location: str = "us-central1",
                 max_retries: int = 3, use_conda: bool = False, environment: str = None, 
                 verbose: bool = False):
        """
        Initialize the SmartInstaller.
        
        Args:
            requirements_file: Path to requirements.txt file
            project_id: Google Cloud project ID (optional, will try to load from credentials)
            location: Google Cloud region for Vertex AI
            max_retries: Maximum number of installation attempts per package
            use_conda: Use conda instead of pip for installations
            environment: Conda environment to use (if use_conda is True)
            verbose: Enable verbose output
        """
        self.requirements_file = requirements_file
        self.project_id = project_id
        self.location = location
        self.max_retries = max_retries
        self.use_conda = use_conda
        self.environment = environment
        self.verbose = verbose
        self.system_info = self._get_system_info()
        self.attempted_solutions = set()
        
        # Try to initialize Google Cloud resources
        self.vertex_ai_available = self._init_google_cloud()

    def _init_google_cloud(self) -> bool:
        """Initialize Google Cloud resources and credentials"""
        # Look for credentials in common locations
        credential_paths = [
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
            os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
            os.path.expanduser("~/gcloud-key.json"),
            os.path.expanduser("~/.gcp/credentials.json"),
            "./service-account.json"
        ]
        
        # Find first valid credentials file
        credentials_path = None
        for path in credential_paths:
            if path and os.path.exists(path):
                credentials_path = path
                break
        
        if credentials_path:
            logger.info(f"Found Google Cloud credentials at: {credentials_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
            # Try to extract project_id from credentials if not provided
            if not self.project_id:
                try:
                    with open(credentials_path, 'r') as f:
                        creds_data = json.load(f)
                        self.project_id = creds_data.get("project_id")
                except Exception as e:
                    logger.warning(f"Could not extract project ID from credentials: {e}")
        
        if not self.project_id:
            # Try to get from gcloud CLI if available
            try:
                self.project_id = subprocess.check_output(
                    ["gcloud", "config", "get-value", "project"], 
                    universal_newlines=True
                ).strip()
            except Exception:
                logger.warning("Could not determine Google Cloud project ID")
        
        # Initialize Vertex AI client
        try:
            if self.project_id:
                # Initialize Vertex AI
                aiplatform.init(project=self.project_id, location=self.location)
                logger.info(f"Initialized Vertex AI with project: {self.project_id}")
                return True
            else:
                logger.warning("No project ID available. Vertex AI features disabled.")
                return False
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI: {e}")
            logger.warning("Using local fallback troubleshooting.")
            return False

    def _get_system_info(self) -> Dict:
        """Gather detailed system information for context"""
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }
        
        # Add more OS-specific details
        if info["os"] == "Linux":
            try:
                # Get Linux distribution info
                import distro
                info["distro"] = distro.id()
                info["distro_version"] = distro.version()
            except ImportError:
                # Fallback if distro module isn't available
                try:
                    with open('/etc/os-release') as f:
                        os_release = {}
                        for line in f:
                            if '=' in line:
                                k, v = line.rstrip().split('=', 1)
                                os_release[k] = v.strip('"')
                    info["distro"] = os_release.get('ID', '')
                    info["distro_version"] = os_release.get('VERSION_ID', '')
                except:
                    info["distro"] = "unknown"
                    info["distro_version"] = "unknown"
                    
            # Check for package managers
            info["has_apt"] = self._command_exists("apt")
            info["has_dnf"] = self._command_exists("dnf")
            info["has_yum"] = self._command_exists("yum")
            
        elif info["os"] == "Darwin":  # macOS
            # Check for macOS package managers
            info["has_homebrew"] = self._command_exists("brew")
            info["has_xcode"] = self._command_exists("xcode-select")
            
            # Get macOS version details
            try:
                mac_ver = platform.mac_ver()
                info["mac_version"] = mac_ver[0]
            except:
                info["mac_version"] = "unknown"
                
        elif info["os"] == "Windows":
            # Check for Windows package managers
            info["has_chocolatey"] = self._command_exists("choco")
            info["has_scoop"] = self._command_exists("scoop")
            info["has_winget"] = self._command_exists("winget")
            info["has_msvc"] = self._check_msvc_installed()
        
        # Get pip or conda version
        if self.use_conda:
            try:
                conda_info = subprocess.check_output(['conda', 'info', '--json'], text=True)
                conda_info = json.loads(conda_info)
                info["package_manager"] = f"conda {conda_info['conda_version']}"
                info["environments"] = conda_info["envs"]
            except Exception as e:
                logger.error(f"Failed to get conda info: {e}")
                info["package_manager"] = "conda (version unknown)"
        else:
            try:
                pip_version = subprocess.check_output([sys.executable, '-m', 'pip', '--version'], text=True)
                info["package_manager"] = pip_version.split()[1]
            except Exception:
                info["package_manager"] = "pip (version unknown)"
        
        return info

    def _command_exists(self, cmd):
        """Check if a command exists on the system"""
        if platform.system() == "Windows":
            try:
                subprocess.check_call(f"where {cmd}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except:
                return False
        else:
            try:
                subprocess.check_call(f"which {cmd}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except:
                return False

    def _check_msvc_installed(self):
        """Check if Microsoft Visual C++ Build Tools are installed"""
        try:
            from setuptools import msvc
            return True
        except:
            try:
                result = subprocess.run("cl.exe", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return result.returncode == 0
            except:
                return False

    def read_requirements(self) -> List[str]:
        """Read requirements from file"""
        if not os.path.exists(self.requirements_file):
            logger.error(f"Requirements file not found: {self.requirements_file}")
            sys.exit(1)
            
        with open(self.requirements_file, 'r') as f:
            requirements = []
            for line in f:
                # Strip comments and whitespace
                line = re.sub(r'#.*$', '', line).strip()
                if line:
                    requirements.append(line)
        
        logger.info(f"Found {len(requirements)} packages in requirements file")
        return requirements

    def install_packages(self) -> bool:
        """Main method to install all packages from requirements"""
        requirements = self.read_requirements()
        
        # Track successfully installed packages
        successful = []
        failed = []
        
        for req in requirements:
            try:
                logger.info(f"Attempting to install {req}")
                success = self.install_package(req)
                if success:
                    successful.append(req)
                else:
                    failed.append(req)
            except KeyboardInterrupt:
                logger.info("Installation interrupted by user")
                break
        
        # Final report
        if successful:
            logger.info(f"Successfully installed {len(successful)} packages: {', '.join(successful)}")
        if failed:
            logger.error(f"Failed to install {len(failed)} packages: {', '.join(failed)}")
        
        return len(failed) == 0

    def install_package(self, package: str) -> bool:
        """Attempt to install a single package with retries and troubleshooting"""
        retries = 0
        
        while retries < self.max_retries:
            cmd = self._build_install_command(package)
            logger.info(f"Running: {' '.join(cmd)}")
            
            try:
                # Run installation command and capture output
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(timeout=600)  # Add 10-minute timeout
                output = stdout + stderr
                
                if self.verbose:
                    logger.debug(f"Command output: {output}")
                
                if process.returncode == 0:
                    logger.info(f"Successfully installed {package}")
                    return True
                    
                logger.warning(f"Installation of {package} failed (attempt {retries+1}/{self.max_retries})")
                logger.warning(f"Error details: {output}")
                
                # Get troubleshooting command
                solution_cmd = self.get_solution(package, output)
                if solution_cmd:
                    logger.info(f"Applying solution: {solution_cmd}")
                    
                    # Execute the solution command with timeout
                    try:
                        # Use non-interactive mode to prevent hanging on prompts
                        env = os.environ.copy()
                        env["DEBIAN_FRONTEND"] = "noninteractive"
                        
                        # Run with timeout to prevent hanging
                        solution_process = subprocess.run(
                            solution_cmd, 
                            shell=True,
                            text=True, 
                            capture_output=True,
                            timeout=300,  # 5-minute timeout
                            env=env,
                            check=False  # Don't raise exception on non-zero return
                        )
                        
                        if solution_process.returncode == 0:
                            logger.info(f"Solution command completed: {solution_process.stdout}")
                        else:
                            logger.warning(f"Solution command failed with code {solution_process.returncode}")
                            logger.warning(f"Error output: {solution_process.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        logger.error(f"Solution command timed out after 5 minutes: {solution_cmd}")
                        # Try to kill the process if it's still running
                        try:
                            import psutil
                            for proc in psutil.process_iter():
                                try:
                                    cmdline = ' '.join(proc.cmdline())
                                    if solution_cmd in cmdline:
                                        logger.warning(f"Killing stalled process: {proc.pid}")
                                        proc.kill()
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except ImportError:
                            logger.warning("psutil not available, cannot kill stalled process")
            except subprocess.TimeoutExpired:
                logger.error(f"Installation command timed out after 10 minutes")
            except Exception as e:
                logger.error(f"Error during installation: {e}")
            
            retries += 1
            time.sleep(2)  # Wait before retrying
        
        logger.error(f"Failed to install {package} after {self.max_retries} attempts")
        return False

    def _build_install_command(self, package: str) -> List[str]:
        """Build the installation command based on package manager and OS"""
        if self.use_conda:
            cmd = ["conda", "install", "-y"]
            if self.environment:
                cmd.extend(["-n", self.environment])
            cmd.append(package)
        else:
            cmd = [sys.executable, "-m", "pip", "install"]
            
            # Add OS-specific pip options
            if self.system_info["os"] == "Linux":
                # On Linux, use --no-cache-dir if we had previous failures with this package
                # as this helps with corrupted cache issues
                if package in self.attempted_solutions:
                    cmd.append("--no-cache-dir")
            
            elif self.system_info["os"] == "Darwin":  # macOS
                # On macOS, use --no-binary for packages that often have issues with wheels
                problem_packages = ["numpy", "scipy", "pandas", "lxml", "pillow"]
                if any(pkg in package.lower() for pkg in problem_packages) and self.system_info.get("has_xcode", False):
                    cmd.append(f"--no-binary={package.split('==')[0]}")
            
            elif self.system_info["os"] == "Windows":
                # On Windows, prefer binary wheels when available
                cmd.append("--prefer-binary")
            
            if self.verbose:
                cmd.append("-v")
                
            cmd.append(package)
        
        return cmd

    def get_solution(self, package: str, error_output: str) -> Optional[str]:
        """Get solution command for installation error with OS awareness"""
        # Try package-specific OS solutions first
        os_solution = self._get_os_specific_solution(package, error_output)
        if os_solution:
            return os_solution
            
        # Then try Vertex AI if available
        if self.vertex_ai_available:
            return self._get_vertex_ai_solution(package, error_output)
        else:
            return self._get_fallback_solution(package, error_output)

    def _get_os_specific_solution(self, package: str, error_output: str) -> Optional[str]:
        """Check for OS-specific installation issues and solutions"""
        os_name = self.system_info["os"]
        base_package = package.split("==")[0].lower() if "==" in package else package.lower()
        
        # List of packages that often need native build dependencies
        build_dependent_packages = [
            "numpy", "scipy", "pandas", "matplotlib", "pillow", "lxml", 
            "cryptography", "pycairo", "pyaudio", "psycopg2", "mysqlclient",
            "h5py", "opencv-python", "dlib"
        ]
        
        # Windows-specific solutions
        if os_name == "Windows":
            if "Microsoft Visual C++" in error_output and "is required" in error_output:
                if not self.system_info.get("has_msvc", False):
                    return "echo \"Installing Microsoft Visual C++ Build Tools - please manually install from https://visualstudio.microsoft.com/visual-cpp-build-tools/\""
                    
            # Check if it's a package that often needs a prebuilt wheel on Windows
            if any(pkg in base_package for pkg in build_dependent_packages):
                return f"{sys.executable} -m pip install --upgrade {base_package} --prefer-binary"
                
            # Add Windows-specific package solutions
            if "dlib" in base_package:
                return f"conda install -c conda-forge dlib || {sys.executable} -m pip install dlib-binary"
                
        # Linux-specific solutions
        elif os_name == "Linux":
            distro = self.system_info.get("distro", "").lower()
            
            # Handle common Linux dependencies
            if any(pkg in base_package for pkg in build_dependent_packages):
                # For Debian/Ubuntu
                if self.system_info.get("has_apt", False):
                    if "numpy" in base_package or "scipy" in base_package:
                        return "sudo apt-get update && sudo apt-get install -y build-essential libatlas-base-dev"
                    elif "pillow" in base_package:
                        return "sudo apt-get update && sudo apt-get install -y libjpeg-dev zlib1g-dev"
                    elif "lxml" in base_package:
                        return "sudo apt-get update && sudo apt-get install -y libxml2-dev libxslt1-dev"
                    elif "opencv" in base_package:
                        return "sudo apt-get update && sudo apt-get install -y libsm6 libxext6 libxrender-dev"
                    elif "dlib" in base_package:
                        return "sudo apt-get update && sudo apt-get install -y cmake"
                    elif "psycopg2" in base_package:
                        return "sudo apt-get update && sudo apt-get install -y libpq-dev"
                
                # For Fedora/RHEL/CentOS
                elif self.system_info.get("has_dnf", False) or self.system_info.get("has_yum", False):
                    pkg_mgr = "dnf" if self.system_info.get("has_dnf", False) else "yum"
                    if "numpy" in base_package or "scipy" in base_package:
                        return f"sudo {pkg_mgr} install -y atlas-devel gcc-c++"
                    elif "pillow" in base_package:
                        return f"sudo {pkg_mgr} install -y libjpeg-devel zlib-devel"
                    elif "lxml" in base_package:
                        return f"sudo {pkg_mgr} install -y libxml2-devel libxslt-devel"
                    elif "opencv" in base_package:
                        return f"sudo {pkg_mgr} install -y libglvnd-glx libsm"
                
        # macOS-specific solutions
        elif os_name == "Darwin":
            if self.system_info.get("has_homebrew", False):
                if "numpy" in base_package or "scipy" in base_package:
                    return "brew install openblas; export OPENBLAS=$(brew --prefix openblas)"
                elif "pillow" in base_package:
                    return "brew install libjpeg zlib"
                elif "lxml" in base_package:
                    return "brew install libxml2 libxslt"
                elif "dlib" in base_package:
                    return "brew install cmake"
                elif "psycopg2" in base_package:
                    return "brew install postgresql"
            
            # Check for Command Line Tools
            if "xcrun" in error_output:
                return "xcode-select --install"
        
        # Try using a different index URL for slow/unavailable packages
        if "failed to download" in error_output.lower() or "timed out" in error_output.lower():
            alt_indices = [
                "--index-url https://pypi.org/simple",
                "--index-url https://pypi.tuna.tsinghua.edu.cn/simple",  # For users in China
                "--index-url https://mirrors.aliyun.com/pypi/simple"     # Another China mirror
            ]
            index = hash(package) % len(alt_indices)
            return f"{sys.executable} -m pip install {alt_indices[index]} {package}"
        
        # No OS-specific solution found
        return None

    def _get_fallback_solution(self, package: str, error_output: str) -> Optional[str]:
        """Fallback error analysis with OS awareness"""
        os_name = self.system_info["os"]
        
        # Common patterns across all platforms
        if "Could not find a version that satisfies the requirement" in error_output:
            # Try with different version constraints
            if "==" in package:
                base_pkg = package.split("==")[0]
                return f"{sys.executable} -m pip install {base_pkg} --no-dependencies"
            # Try a different index
            else:
                return f"{sys.executable} -m pip install {package} --index-url https://pypi.org/simple"
        
        # OS-specific patterns
        if os_name == "Windows":
            if "Microsoft Visual C++" in error_output:
                return "echo \"You need Microsoft Visual C++ Build Tools. Please install from https://visualstudio.microsoft.com/visual-cpp-build-tools/\""
            elif "No matching distribution found" in error_output:
                return f"{sys.executable} -m pip install {package} --prefer-binary"
            elif "Access is denied" in error_output or "Permission denied" in error_output:
                # Try running as admin
                return f"echo \"Try running as administrator: Right-click Command Prompt and select 'Run as administrator'\""
                
        elif os_name == "Linux":
            if "Permission denied" in error_output:
                return f"{sys.executable} -m pip install {package} --user"
            elif "fatal error: Python.h: No such file or directory" in error_output:
                if self.system_info.get("has_apt", False):
                    return f"sudo apt-get update && sudo apt-get install -y python{sys.version_info.major}.{sys.version_info.minor}-dev"
                elif self.system_info.get("has_dnf", False):
                    return f"sudo dnf install -y python{sys.version_info.major}.{sys.version_info.minor}-devel"
                elif self.system_info.get("has_yum", False):
                    return f"sudo yum install -y python{sys.version_info.major}.{sys.version_info.minor}-devel"
                    
        elif os_name == "Darwin":  # macOS
            if "Failed to build" in error_output:
                if not self.system_info.get("has_xcode", True):
                    return "xcode-select --install"
                else:
                    return f"{sys.executable} -m pip install {package} --no-binary=:all:"
                    
        # Common fallbacks for all platforms
        if "Failed building wheel" in error_output:
            return f"{sys.executable} -m pip install wheel setuptools --upgrade"
            
        if "SSLError" in error_output or "CERTIFICATE_VERIFY_FAILED" in error_output:
            return f"{sys.executable} -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org {package}"
        
        # No fallback solution
        return None

    def _get_vertex_ai_solution(self, package: str, error_output: str) -> Optional[str]:
        """Use Vertex AI to analyze error and suggest a solution command"""
        try:
            # Create more detailed system information for the AI
            system_info_str = '\n'.join([f"{k}: {v}" for k, v in self.system_info.items()])
            
            # Provide OS-specific hints to the AI
            os_specific_hints = ""
            if self.system_info["os"] == "Windows":
                os_specific_hints = """
                - Windows often needs Visual C++ Build Tools for compiling
                - Consider using --prefer-binary option
                - Check if user has admin rights
                """
            elif self.system_info["os"] == "Linux":
                os_specific_hints = f"""
                - This is a {self.system_info.get('distro', 'Linux')} system
                - Consider recommending appropriate system packages
                - User may need to use sudo for system dependencies
                """
            elif self.system_info["os"] == "Darwin":  # macOS
                os_specific_hints = """
                - This is a macOS system
                - Check if XCode Command Line Tools are installed
                - Consider using Homebrew for dependencies
                """
            
            # Create enhanced prompt for the AI
            prompt = f"""
            You are an expert Python package installation troubleshooter specialized in {self.system_info['os']} environments.
            Analyze the installation error and provide a SINGLE shell command that will fix the issue.
            Return ONLY the command with no explanation or markdown formatting.
            
            System Information:
            {system_info_str}
            
            OS-Specific Considerations:
            {os_specific_hints}
            
            Package being installed: {package}
            Installation tool: {"conda" if self.use_conda else "pip"}
            
            Error output:
            {error_output}
            
            Return ONLY a shell command that would likely fix the issue. If multiple solutions are possible, 
            pick the most likely to succeed. If you cannot determine a solution, return "NONE".
            """
            
            # Get response from Vertex AI
            model = GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            
            # Extract command from response
            solution_cmd = response.text.strip()
            
            # Ignore solutions we've already tried
            if solution_cmd in self.attempted_solutions:
                logger.info(f"Ignoring previously attempted solution: {solution_cmd}")
                return None
            
            self.attempted_solutions.add(solution_cmd)
            
            # Don't return "NONE" as a command
            if solution_cmd == "NONE":
                return None
                
            return solution_cmd
            
        except Exception as e:
            logger.error(f"Error using Vertex AI: {e}")
            return self._get_fallback_solution(package, error_output)

def main():
    parser = argparse.ArgumentParser(description='Smart Package Installer')
    parser.add_argument('requirements', nargs='?', default='requirements.txt',
                      help='Path to requirements.txt file (default: requirements.txt)')
    parser.add_argument('--project-id', help='Google Cloud project ID')
    parser.add_argument('--location', default='us-central1', 
                      help='Google Cloud region (default: us-central1)')
    parser.add_argument('--conda', action='store_true', help='Use conda instead of pip')
    parser.add_argument('--env', help='Conda environment name (if using conda)')
    parser.add_argument('--retries', type=int, default=3, help='Maximum installation retries per package')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    installer = SmartInstaller(
        requirements_file=args.requirements,
        project_id=args.project_id,
        location=args.location,
        max_retries=args.retries,
        use_conda=args.conda,
        environment=args.env,
        verbose=args.verbose
    )
    
    success = installer.install_packages()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
