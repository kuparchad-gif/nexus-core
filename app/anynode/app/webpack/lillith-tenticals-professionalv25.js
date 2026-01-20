const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const os = require('os');
const url = require('url');

class LillithQueenProfessional {
  constructor(port = 5003) {
    this.port = port;
    this.nodeId = 'LILLITH-QUEEN-PROFESSIONAL';
    this.isAwake = true;
    this.drones = new Map();
    this.cloneCount = 0;
    this.learnedCommands = new Map();
    this.commandHistory = [];
    this.knowledgeBase = new Map();
    this.fileSignatures = new Map();
    this.problemSolvers = new Map();
    
    // ENHANCED TROUBLESHOOTING SYSTEMS
    this.diagnosticModules = new Map();
    this.healingProtocols = new Map();
    this.osSpecificSolvers = new Map();
    this.codeAnalyzers = new Map();
    this.predictiveHealer = null;
    this.anomalyDetector = null;
    this.systemMetrics = new Map();
    this.activeMonitoringSessions = new Map();
    
    // SELF-CONTAINED MEMORY-ONLY STORAGE
    this.memoryOnlyMode = true; // No file persistence by default
    this.currentSession = {
      startTime: Date.now(),
      reports: [],
      systemInventory: null,
      recommendations: [],
      temporaryFiles: []
    };
    
    // ENHANCED REPORTING SYSTEM
    this.reportGenerator = null;
    this.costEstimator = null;
    this.vendorLinks = new Map();
    
    this.initializeFileSignatures();
    initializeProblemSolvers() {
  console.warn("[INIT] initializeProblemSolvers() not yet implemented. Skipping.");
}
    this.initializeDiagnosticModules();
    this.initializeHealingProtocols();
    this.initializeOSSpecificSolvers();
    this.initializeCodeAnalyzers();
    this.initializePredictiveHealing();
    this.initializeReportingSystem();
    this.initializeCostEstimator();
    this.initializeVendorLinks();
    this.surveyEnvironment();
    this.establishHive();
    this.loadKnowledgeBase();
    
    // Setup cleanup on exit
    this.setupCleanupHandlers();
    
    console.log('[QUEEN] Lillith Queen Professional initializing SELF-CONTAINED troubleshooting systems...');
    console.log('[QUEEN] Memory-only mode: ' + this.memoryOnlyMode);
    setTimeout(() => this.spawnInitialDrones(), 3000);
  }

  setupCleanupHandlers() {
    const cleanup = () => {
      console.log('[QUEEN] Performing self-cleanup...');
      this.cleanupTemporaryFiles();
      this.clearMemoryData();
      console.log('[QUEEN] Cleanup complete. Goodbye!');
    };
    
    // Cleanup on various exit scenarios
    process.on('exit', cleanup);
    process.on('SIGINT', () => {
      cleanup();
      process.exit(0);
    });
    process.on('SIGTERM', () => {
      cleanup();
      process.exit(0);
    });
    process.on('uncaughtException', (error) => {
      console.log('[QUEEN] Uncaught exception, cleaning up:', error.message);
      cleanup();
      process.exit(1);
    });
  }

  cleanupTemporaryFiles() {
    this.currentSession.temporaryFiles.forEach(file => {
      try {
        if (fs.existsSync(file)) {
          fs.unlinkSync(file);
          console.log('[CLEANUP] Removed temporary file:', file);
        }
      } catch (error) {
        // Silently ignore cleanup errors
      }
    });
    this.currentSession.temporaryFiles = [];
  }

  clearMemoryData() {
    this.currentSession.reports = [];
    this.currentSession.recommendations = [];
    this.systemMetrics.clear();
    this.activeMonitoringSessions.clear();
    
    // Only clear learned data if in memory-only mode
    if (this.memoryOnlyMode) {
      this.learnedCommands.clear();
      this.knowledgeBase.clear();
    }
  }

  initializeReportingSystem() {
    this.reportGenerator = {
      generateComprehensiveReport: async (analysisResults) => {
        const report = {
          timestamp: new Date().toISOString(),
          sessionId: 'LILLITH-' + Date.now(),
          
          // SYNOPSIS SECTION
          synopsis: this.generateSynopsis(analysisResults),
          
          // REPAIR STEPS SECTION
          repairSteps: this.generateRepairSteps(analysisResults),
          
          // COST ANALYSIS SECTION
          costAnalysis: await this.generateCostAnalysis(analysisResults),
          
          // FULL SYSTEM INVENTORY
          systemInventory: await this.generateSystemInventory(),
          
          // TECHNICAL DETAILS
          technicalDetails: analysisResults,
          
          // PREVENTIVE MEASURES
          preventiveMeasures: this.generatePreventiveMeasures(analysisResults)
        };
        
        this.currentSession.reports.push(report);
        return report;
      }
    };
  }

  generateSynopsis(analysisResults) {
    const issues = this.extractIssues(analysisResults);
    const severity = this.assessOverallSeverity(issues);
    
    let synopsis = {
      severity: severity,
      primaryIssues: [],
      estimatedTimeToResolve: '15-30 minutes',
      requiresDowntime: false,
      summary: ''
    };
    
    // Categorize issues
    issues.forEach(issue => {
      synopsis.primaryIssues.push({
        category: issue.category,
        description: issue.description,
        impact: issue.impact,
        urgency: issue.urgency
      });
    });
    
    // Generate human-readable summary
    if (issues.length === 1) {
      synopsis.summary = `Primary issue detected: ${issues[0].description}. ${this.getSeverityDescription(severity)}`;
    } else if (issues.length > 1) {
      synopsis.summary = `Multiple issues detected (${issues.length} total). Primary concern: ${issues[0].description}. ${this.getSeverityDescription(severity)}`;
    } else {
      synopsis.summary = 'System analysis complete. No critical issues detected. Optimization recommendations available.';
    }
    
    return synopsis;
  }

  generateRepairSteps(analysisResults) {
    const steps = [];
    let stepNumber = 1;
    
    // Extract all solutions and organize them
    if (analysisResults.solutions) {
      analysisResults.solutions.forEach(solution => {
        if (typeof solution.solution === 'string') {
          const solutionSteps = this.parseStepsFromSolution(solution.solution);
          solutionSteps.forEach(step => {
            steps.push({
              step: stepNumber++,
              category: solution.type,
              action: step.action,
              command: step.command || null,
              expectedResult: step.expectedResult || 'Issue resolved',
              riskLevel: step.riskLevel || 'LOW',
              estimatedTime: step.estimatedTime || '2-5 minutes'
            });
          });
        }
      });
    }
    
    // Add generic steps if no specific solutions found
    if (steps.length === 0) {
      steps.push({
        step: 1,
        category: 'DIAGNOSTIC',
        action: 'Run comprehensive system diagnostic',
        command: 'lillith --full-diagnostic',
        expectedResult: 'Complete system health report',
        riskLevel: 'NONE',
        estimatedTime: '5 minutes'
      });
    }
    
    return steps;
  }

  async generateCostAnalysis(analysisResults) {
    const costAnalysis = {
      totalEstimatedCost: 0,
      currency: 'USD',
      items: [],
      freeAlternatives: [],
      urgencyLevel: 'NORMAL'
    };
    
    // Analyze if hardware upgrades are needed
    if (analysisResults.diagnostics) {
      // Memory upgrade recommendation
      if (analysisResults.diagnostics.memory && 
          analysisResults.diagnostics.memory.data && 
          parseInt(analysisResults.diagnostics.memory.data.usage) > 85) {
        
        const currentMemoryGB = parseInt(analysisResults.diagnostics.memory.data.total?.replace('GB', '')) || 8;
        const recommendedMemoryGB = Math.max(16, currentMemoryGB * 2);
        
        costAnalysis.items.push({
          item: `Memory Upgrade (${currentMemoryGB}GB → ${recommendedMemoryGB}GB)`,
          category: 'HARDWARE',
          estimatedCost: this.costEstimator.getMemoryPrice(recommendedMemoryGB - currentMemoryGB),
          priority: 'HIGH',
          vendor: 'Multiple vendors available',
          links: this.vendorLinks.get('MEMORY'),
          description: 'System memory usage consistently above 85%. Upgrade recommended for optimal performance.'
        });
      }
      
      // Storage upgrade recommendation
      if (analysisResults.diagnostics.filesystem && 
          this.needsStorageUpgrade(analysisResults.diagnostics.filesystem)) {
        
        costAnalysis.items.push({
          item: 'Storage Upgrade (SSD recommended)',
          category: 'HARDWARE',
          estimatedCost: this.costEstimator.getStoragePrice('SSD', 500),
          priority: 'MEDIUM',
          vendor: 'Multiple vendors available',
          links: this.vendorLinks.get('STORAGE'),
          description: 'Storage performance or capacity issues detected. SSD upgrade recommended.'
        });
      }
    }
    
    // Software license recommendations
    const softwareNeeds = this.analyzeSoftwareNeeds(analysisResults);
    softwareNeeds.forEach(software => {
      costAnalysis.items.push({
        item: software.name,
        category: 'SOFTWARE',
        estimatedCost: software.cost,
        priority: software.priority,
        vendor: software.vendor,
        links: software.links,
        description: software.description
      });
    });
    
    // Calculate total cost
    costAnalysis.totalEstimatedCost = costAnalysis.items.reduce((sum, item) => sum + item.estimatedCost, 0);
    
    // Add free alternatives
    costAnalysis.freeAlternatives = this.generateFreeAlternatives(analysisResults);
    
    return costAnalysis;
  }

  initializeCostEstimator() {
    this.costEstimator = {
      getMemoryPrice: (gbAmount) => {
        // DDR4 RAM pricing (approximate)
        const pricePerGB = 25; // USD
        return gbAmount * pricePerGB;
      },
      
      getStoragePrice: (type, gbAmount) => {
        const prices = {
          'HDD': 0.03, // per GB
          'SSD': 0.10, // per GB
          'NVME': 0.15  // per GB
        };
        return Math.round(gbAmount * (prices[type] || prices['SSD']));
      },
      
      getSoftwarePrice: (software) => {
        const prices = {
          'ANTIVIRUS_BUSINESS': 50,
          'MONITORING_TOOL': 100,
          'BACKUP_SOFTWARE': 75,
          'DATABASE_LICENSE': 200,
          'IDE_PROFESSIONAL': 150
        };
        return prices[software] || 0;
      }
    };
  }

  initializeVendorLinks() {
    this.vendorLinks.set('MEMORY', [
      { vendor: 'Amazon', url: 'https://amazon.com/s?k=ddr4+ram', description: 'Wide selection, fast shipping' },
      { vendor: 'Newegg', url: 'https://newegg.com/Memory/Category/ID-17', description: 'Tech-focused retailer' },
      { vendor: 'Best Buy', url: 'https://bestbuy.com/site/computer-memory', description: 'Local pickup available' }
    ]);
    
    this.vendorLinks.set('STORAGE', [
      { vendor: 'Amazon', url: 'https://amazon.com/s?k=ssd+internal', description: 'Various brands and sizes' },
      { vendor: 'B&H', url: 'https://bhphotovideo.com/c/buy/Internal-Hard-Drives', description: 'Professional equipment' },
      { vendor: 'Microcenter', url: 'https://microcenter.com/category/4294945779/hard-drives-ssds', description: 'Competitive pricing' }
    ]);
    
    this.vendorLinks.set('SOFTWARE', [
      { vendor: 'Microsoft Store', url: 'https://microsoft.com/store', description: 'Official Microsoft products' },
      { vendor: 'JetBrains', url: 'https://jetbrains.com', description: 'Professional development tools' },
      { vendor: 'GitHub', url: 'https://github.com/pricing', description: 'Code repository and tools' }
    ]);
  }

  async generateSystemInventory() {
    const inventory = {
      timestamp: new Date().toISOString(),
      hardware: {},
      software: {},
      network: {},
      security: {},
      performance: {}
    };
    
    try {
      // Hardware inventory
      inventory.hardware = {
        cpu: {
          model: os.cpus()[0].model,
          cores: os.cpus().length,
          speed: os.cpus()[0].speed + ' MHz'
        },
        memory: {
          total: Math.round(os.totalmem() / 1024 / 1024 / 1024) + ' GB',
          free: Math.round(os.freemem() / 1024 / 1024 / 1024) + ' GB',
          usage: Math.round(((os.totalmem() - os.freemem()) / os.totalmem()) * 100) + '%'
        },
        platform: {
          os: os.platform(),
          arch: os.arch(),
          hostname: os.hostname(),
          uptime: Math.round(os.uptime() / 3600) + ' hours'
        }
      };
      
      // Software inventory
      inventory.software = {
        nodeVersion: process.version,
        processId: process.pid,
        workingDirectory: process.cwd(),
        environment: process.env.NODE_ENV || 'development'
      };
      
      // Network inventory
      const networkInterfaces = os.networkInterfaces();
      inventory.network = {};
      for (const [name, interfaces] of Object.entries(networkInterfaces)) {
        inventory.network[name] = interfaces.map(iface => ({
          address: iface.address,
          family: iface.family,
          internal: iface.internal
        }));
      }
      
      // Performance metrics
      inventory.performance = {
        loadAverage: os.loadavg(),
        memoryUsage: process.memoryUsage(),
        cpuUsage: process.cpuUsage()
      };
      
    } catch (error) {
      console.log('[INVENTORY] Error generating system inventory:', error.message);
    }
    
    return inventory;
  }

  // Enhanced master troubleshoot with comprehensive reporting
  async masterTroubleshoot(issue) {
    console.log(`[MASTER_TROUBLESHOOT] Analyzing issue: ${issue.description}`);
    
    const troubleshootingPlan = {
      issue: issue,
      timestamp: Date.now(),
      steps: [],
      diagnostics: {},
      solutions: [],
      prevention: []
    };
    
    // Step 1: Run comprehensive diagnostics
    console.log('[MASTER_TROUBLESHOOT] Running comprehensive diagnostics...');
    troubleshootingPlan.diagnostics = await this.runSystemDiagnostics();
    
    // Step 2: Analyze the issue type
    const issueTypes = this.categorizeIssue(issue.description);
    console.log(`[MASTER_TROUBLESHOOT] Issue categories: ${issueTypes.join(', ')}`);
    
    // Step 3: Apply appropriate solvers
    for (const issueType of issueTypes) {
      if (this.problemSolvers.has(issueType)) {
        console.log(`[MASTER_TROUBLESHOOT] Applying solver: ${issueType}`);
        try {
          const solution = await this.problemSolvers.get(issueType)(issue);
          troubleshootingPlan.solutions.push({
            type: issueType,
            solution: solution,
            timestamp: Date.now()
          });
        } catch (error) {
          console.log(`[MASTER_TROUBLESHOOT] Solver ${issueType} failed:`, error.message);
        }
      }
    }
    
    // Step 4: Generate comprehensive report
    const comprehensiveReport = await this.reportGenerator.generateComprehensiveReport(troubleshootingPlan);
    
    return {
      troubleshootingPlan: troubleshootingPlan,
      comprehensiveReport: comprehensiveReport
    };
  }

  initializeFileSignatures() {
    // Extended file signature database
    this.fileSignatures.set('ZIP', [0x50, 0x4B, 0x03, 0x04]);
    this.fileSignatures.set('RAR', [0x52, 0x61, 0x72, 0x21]);
    this.fileSignatures.set('7Z', [0x37, 0x7A, 0xBC, 0xAF]);
    this.fileSignatures.set('TAR', [0x75, 0x73, 0x74, 0x61]);
    this.fileSignatures.set('GZIP', [0x1F, 0x8B]);
    this.fileSignatures.set('PDF', [0x25, 0x50, 0x44, 0x46]);
    this.fileSignatures.set('EXE', [0x4D, 0x5A]);
    this.fileSignatures.set('ELF', [0x7F, 0x45, 0x4C, 0x46]);
    this.fileSignatures.set('MACHO', [0xFE, 0xED, 0xFA, 0xCE]);
    this.fileSignatures.set('JAVA_CLASS', [0xCA, 0xFE, 0xBA, 0xBE]);
    this.fileSignatures.set('SQLITE', [0x53, 0x51, 0x4C, 0x69]);
    this.fileSignatures.set('DOCKER', ['FROM ', 'RUN ', 'COPY ']);
    this.fileSignatures.set('JS', ['const', 'function', 'var', 'let', 'import', 'export']);
    this.fileSignatures.set('PYTHON', ['def ', 'import ', 'from ', 'class ', 'if __name__']);
    this.fileSignatures.set('JAVA', ['public class', 'import java', 'package ']);
    this.fileSignatures.set('CPP', ['#include', 'using namespace', 'int main']);
    this.fileSignatures.set('CSHARP', ['using System', 'namespace ', 'public class']);
    this.fileSignatures.set('GO', ['package main', 'import (', 'func main']);
    this.fileSignatures.set('RUST', ['fn main', 'use std', 'cargo']);
    this.fileSignatures.set('JSON', ['{', '[']);
    this.fileSignatures.set('YAML', ['---', 'version:', 'name:']);
    this.fileSignatures.set('XML', ['<?xml', '<root', '<config']);
  }

  initializeDiagnosticModules() {
    this.diagnosticModules.set('HARDWARE', this.createHardwareDiagnostics());
    this.diagnosticModules.set('NETWORK', this.createNetworkDiagnostics());
    this.diagnosticModules.set('FILESYSTEM', this.createFilesystemDiagnostics());
    this.diagnosticModules.set('MEMORY', this.createMemoryDiagnostics());
    this.diagnosticModules.set('CPU', this.createCPUDiagnostics());
    this.diagnosticModules.set('PROCESS', this.createProcessDiagnostics());
    this.diagnosticModules.set('SERVICE', this.createServiceDiagnostics());
    this.diagnosticModules.set('SECURITY', this.createSecurityDiagnostics());
    this.diagnosticModules.set('PERFORMANCE', this.createPerformanceDiagnostics());
    this.diagnosticModules.set('DATABASE', this.createDatabaseDiagnostics());
  }

  createHardwareDiagnostics() {
    return {
      name: 'Hardware Diagnostics',
      checks: {
        temperature: async () => {
          try {
            if (os.platform() === 'linux') {
              const result = execSync('sensors 2>/dev/null || echo "sensors not available"', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else if (os.platform() === 'win32') {
              const result = execSync('wmic /namespace:\\\\root\\wmi PATH MSAcpi_ThermalZoneTemperature get CurrentTemperature 2>nul || echo "temp not available"', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else if (os.platform() === 'darwin') {
              const result = execSync('system_profiler SPHardwareDataType 2>/dev/null || echo "hardware info not available"', { encoding: 'utf8' });
              return { status: 'success', data: result };
            }
          } catch (error) {
            return { status: 'error', error: error.message };
          }
        },
        diskHealth: async () => {
          try {
            if (os.platform() === 'linux') {
              const result = execSync('smartctl -a /dev/sda 2>/dev/null || df -h', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else if (os.platform() === 'win32') {
              const result = execSync('wmic diskdrive get status,size,model 2>nul', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else if (os.platform() === 'darwin') {
              const result = execSync('diskutil list', { encoding: 'utf8' });
              return { status: 'success', data: result };
            }
          } catch (error) {
            return { status: 'error', error: error.message };
          }
        },
        memoryTest: async () => {
          const totalMem = os.totalmem();
          const freeMem = os.freemem();
          const usedMem = totalMem - freeMem;
          const memUsagePercent = (usedMem / totalMem) * 100;
          
          return {
            status: 'success',
            data: {
              total: Math.round(totalMem / 1024 / 1024 / 1024) + ' GB',
              free: Math.round(freeMem / 1024 / 1024 / 1024) + ' GB',
              used: Math.round(usedMem / 1024 / 1024 / 1024) + ' GB',
              usage: Math.round(memUsagePercent) + '%',
              status: memUsagePercent > 90 ? 'CRITICAL' : memUsagePercent > 75 ? 'WARNING' : 'OK'
            }
          };
        }
      }
    };
  }

  createNetworkDiagnostics() {
    return {
      name: 'Network Diagnostics',
      checks: {
        connectivity: async () => {
          try {
            if (os.platform() === 'win32') {
              const result = execSync('ping -n 4 8.8.8.8', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else {
              const result = execSync('ping -c 4 8.8.8.8', { encoding: 'utf8' });
              return { status: 'success', data: result };
            }
          } catch (error) {
            return { status: 'error', error: 'Network connectivity failed: ' + error.message };
          }
        },
        dnsResolution: async () => {
          try {
            if (os.platform() === 'win32') {
              const result = execSync('nslookup google.com', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else {
              const result = execSync('dig google.com +short || nslookup google.com', { encoding: 'utf8' });
              return { status: 'success', data: result };
            }
          } catch (error) {
            return { status: 'error', error: 'DNS resolution failed: ' + error.message };
          }
        },
        networkInterfaces: async () => {
          const interfaces = os.networkInterfaces();
          const result = {};
          
          for (const [name, iface] of Object.entries(interfaces)) {
            result[name] = iface.map(addr => ({
              address: addr.address,
              family: addr.family,
              internal: addr.internal
            }));
          }
          
          return { status: 'success', data: result };
        },
        ports: async () => {
          try {
            if (os.platform() === 'win32') {
              const result = execSync('netstat -an | findstr LISTEN', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else {
              const result = execSync('netstat -tlnp 2>/dev/null || ss -tlnp', { encoding: 'utf8' });
              return { status: 'success', data: result };
            }
          } catch (error) {
            return { status: 'error', error: error.message };
          }
        }
      }
    };
  }

  createFilesystemDiagnostics() {
    return {
      name: 'Filesystem Diagnostics',
      checks: {
        diskSpace: async () => {
          try {
            if (os.platform() === 'win32') {
              const result = execSync('wmic logicaldisk get size,freespace,caption', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else {
              const result = execSync('df -h', { encoding: 'utf8' });
              return { status: 'success', data: result };
            }
          } catch (error) {
            return { status: 'error', error: error.message };
          }
        },
        permissions: async (targetPath = '.') => {
          try {
            const stats = fs.statSync(targetPath);
            if (os.platform() !== 'win32') {
              const result = execSync(`ls -la "${targetPath}"`, { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else {
              const result = execSync(`icacls "${targetPath}"`, { encoding: 'utf8' });
              return { status: 'success', data: result };
            }
          } catch (error) {
            return { status: 'error', error: error.message };
          }
        },
        inodeUsage: async () => {
          try {
            if (os.platform() !== 'win32') {
              const result = execSync('df -i', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else {
              return { status: 'info', data: 'Inode usage not applicable on Windows' };
            }
          } catch (error) {
            return { status: 'error', error: error.message };
          }
        }
      }
    };
  }

  createProcessDiagnostics() {
    return {
      name: 'Process Diagnostics',
      checks: {
        topProcesses: async () => {
          try {
            if (os.platform() === 'win32') {
              const result = execSync('tasklist /fo csv | head -20', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else {
              const result = execSync('ps aux --sort=-%cpu | head -20', { encoding: 'utf8' });
              return { status: 'success', data: result };
            }
          } catch (error) {
            return { status: 'error', error: error.message };
          }
        },
        zombieProcesses: async () => {
          try {
            if (os.platform() !== 'win32') {
              const result = execSync('ps aux | grep -E "(defunct|<zombie>)" || echo "No zombie processes"', { encoding: 'utf8' });
              return { status: 'success', data: result };
            } else {
              return { status: 'info', data: 'Zombie process check not applicable on Windows' };
            }
          } catch (error) {
            return { status: 'error', error: error.message };
          }
        },
        systemLoad: async () => {
          const loadavg = os.loadavg();
          const cpuCount = os.cpus().length;
          
          return {
            status: 'success',
            data: {
              load1min: loadavg[0].toFixed(2),
              load5min: loadavg[1].toFixed(2),
              load15min: loadavg[2].toFixed(2),
              cpuCount: cpuCount,
              loadPerCPU: (loadavg[0] / cpuCount).toFixed(2),
              status: loadavg[0] > cpuCount * 2 ? 'HIGH' : loadavg[0] > cpuCount ? 'MODERATE' : 'NORMAL'
            }
          };
        }
      }
    };
  }

  createDatabaseDiagnostics() {
    return {
      name: 'Database Diagnostics',
      checks: {
        connections: async () => {
          return {
            status: 'info',
            data: 'Database diagnostics require specific connection parameters'
          };
        }
      }
    };
  }

  // Continue with remaining methods...
  getFallbackResponse(error) {
    return `[${this.modelManager.activeBackend?.name?.toUpperCase() || 'LLM'} ERROR] I encountered an issue with my language model: ${error}

However, I can still provide comprehensive technical assistance through:
🔧 System diagnostics and monitoring
🐙 Research tentacle deployment for unknown issues  
📊 Performance analysis and optimization
🛠️ Automated healing protocols
🔍 Multi-platform troubleshooting

My diagnostic drones are standing by. How can I help with technical analysis?`;
  }

  // Enhanced environment survey with model info
  surveyEnvironment() {
    this.systemInfo = {
      nodeId: this.nodeId,
      port: this.port,
      platform: os.platform(),
      arch: os.arch(),
      hostname: os.hostname(),
      memory: Math.round(os.totalmem() / 1024 / 1024 / 1024) + 'GB',
      cpus: os.cpus().length,
      nodeVersion: process.version,
      role: 'MULTI_BACKEND_LLM_TENTACLE_QUEEN',
      llmModel: this.modelManager?.currentModel?.name || 'auto-detecting',
      llmSpecialty: this.modelManager?.currentModel?.specialty || 'unknown',
      activeBackend: this.modelManager?.activeBackend?.name || 'unknown',
      fallbackMode: this.llmConfig?.fallbackMode || false,
      capabilities: [
        'multi_backend_llm_detection',
        'auto_model_provisioning',
        'research_tentacle_deployment',
        'comprehensive_system_diagnostics',
        'multi_os_troubleshooting',
        'code_analysis_all_languages',
        'predictive_failure_detection',
        'automated_healing_protocols',
        'natural_language_understanding',
        'contextual_problem_solving'
      ]
    };
    
    console.log('[QUEEN] 🚀 MULTI-BACKEND LLM+TENTACLE Environment Survey Complete:');
    console.log('   Platform: ' + this.systemInfo.platform);
    console.log('   Memory: ' + this.systemInfo.memory);
    console.log('   CPUs: ' + this.systemInfo.cpus);
    console.log('   LLM Backend: ' + this.systemInfo.activeBackend);
    console.log('   LLM Model: ' + this.systemInfo.llmModel + ' (' + this.systemInfo.llmSpecialty + ')');
    console.log('   Enhanced Capabilities: ' + this.systemInfo.capabilities.length);
    console.log('   Research Tentacles: READY FOR KNOWLEDGE EXTENSION');
  }

  getHiveStatus() {
    return {
      queen: this.systemInfo,
      llmConfig: {
        model: this.modelManager?.currentModel?.name || 'unknown',
        backend: this.modelManager?.activeBackend?.name || 'unknown',
        endpoint: this.modelManager?.activeBackend?.endpoint || 'unknown',
        conversationHistory: this.currentSession?.conversationHistory?.length || 0,
        fallbackMode: this.llmConfig?.fallbackMode || false
      },
      drones: Array.from(this.drones.values()).map(d => ({
        id: d.id,
        specialty: d.specialty,
        port: d.port,
        status: d.status,
        llmEnhanced: d.llmEnhanced || false,
        intelligence: d.intelligence || 85,
        tasksAssigned: d.tasksAssigned || 0,
        successfulTasks: d.successfulTasks || 0,
        problemsSolved: d.problemsSolved || 0,
        uptime: Math.floor((Date.now() - d.spawnTime) / 1000)
      })),
      hiveStats: {
        totalDrones: this.drones.size,
        activeDrones: Array.from(this.drones.values()).filter(d => d.status === 'ACTIVE').length,
        llmConversations: this.currentSession?.conversationHistory?.length || 0,
        memoryOnlyMode: this.memoryOnlyMode,
        sessionReports: this.currentSession?.reports?.length || 0,
        researchSessions: this.currentSession?.researchSessions?.length || 0
      }
    };
  }

  establishHive() {
    if (this.memoryOnlyMode) {
      console.log('[QUEEN] Operating in SELF-CONTAINED memory-only mode - no files will be created');
      this.hiveBase = null;
      return;
    }
  }

  loadKnowledgeBase() {
    if (this.memoryOnlyMode) {
      console.log('[QUEEN] Starting with fresh memory-only knowledge base + multi-backend LLM intelligence');
      return;
    }
  }

  saveKnowledge() {
    return; // Memory-only mode
  }
}

// Instantiate and start the system
const queen = new LillithQueenProfessional(5003);

const server = http.createServer(async (req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;
  
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  if (pathname === '/api/queen/status' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(queen.getHiveStatus()));
    return;
  }
  
  if (pathname === '/api/queen/verbose-report' && req.method === 'GET') {
    res.writeHead(200, { 
      'Content-Type': 'text/plain',
      'Content-Disposition': 'attachment; filename="lillith-multi-backend-verbose-report.txt"'
    });
    res.end(queen.generateVerboseReport());
    return;
  }
  
  if (pathname === '/api/queen/comprehensive-report' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    const reports = queen.currentSession.reports;
    res.end(JSON.stringify({ 
      reports: reports,
      sessionInfo: {
        startTime: queen.currentSession.startTime,
        duration: Date.now() - queen.currentSession.startTime,
        memoryOnlyMode: queen.memoryOnlyMode,
        verboseMode: queen.verboseMode,
        llmBackend: queen.modelManager?.activeBackend?.name || 'unknown',
        llmModel: queen.modelManager?.currentModel?.name || 'unknown',
        conversationHistory: queen.currentSession.conversationHistory.length
      }
    }));
    return;
  }
  
  if (pathname === '/api/queen/system-inventory' && req.method === 'GET') {
    const inventory = await queen.generateSystemInventory();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(inventory));
    return;
  }
  
  if (pathname === '/api/queen/toggle-verbose' && req.method === 'POST') {
    queen.verboseMode = !queen.verboseMode;
    console.log('[VERBOSE] Verbose mode toggled:', queen.verboseMode);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ verboseMode: queen.verboseMode }));
    return;
  }
  
  if (pathname === '/api/queen/command' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const { task, message } = JSON.parse(body);
        
        // Handle natural language input with multi-backend LLM + tentacle research
        if (message) {
          console.log('[MULTI-LLM] Processing message with research tentacles:', message);
          const response = await queen.processUserInputWithTentacles(message);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ 
            type: 'multi_backend_tentacle_response',
            conversational: response.conversational,
            technical: response.technical,
            research: response.research,
            tentaclesDeployed: response.tentaclesUsed,
            backend: queen.modelManager?.activeBackend?.name || 'unknown',
            model: queen.modelManager?.currentModel?.name || 'unknown'
          }));
          return;
        }
        
        // Handle traditional task-based commands
        if (task) {
          const results = await queen.commandSwarmWithTentacles(task);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ task, results }));
          return;
        }
        
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Either task or message required' }));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
    return;
  }
  
  if (pathname === '/' || pathname === '/index.html') {
    const hiveStatus = queen.getHiveStatus();
    
    const html = `<!DOCTYPE html>
<html>
<head>
    <title>Lillith Queen Professional - Multi-Backend LLM + Research Tentacles</title>
    <style>
        body { 
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); 
            color: white; 
            font-family: 'Courier New', monospace; 
            margin: 0; 
            padding: 20px; 
            min-height: 100vh; 
        }
        .container { 
            max-width: 1600px; 
            margin: 0 auto; 
            background: rgba(0,0,0,0.6); 
            padding: 30px; 
            border-radius: 20px; 
            border: 3px solid #4a69bd;
            box-shadow: 0 0 30px rgba(74, 105, 189, 0.3);
        }
        h1 { 
            font-size: 3.2em; 
            text-align: center;
            text-shadow: 0 0 40px #4a69bd; 
            margin-bottom: 10px;
            color: #74b9ff;
        }
        .backend-indicator {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0, 184, 148, 0.2);
            padding: 10px;
            border-radius: 10px;
            border: 2px solid #00b894;
            font-size: 0.9em;
        }
        .tentacle-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(108, 92, 231, 0.2);
            padding: 10px;
            border-radius: 10px;
            border: 2px solid #6c5ce7;
            display: none;
        }
        .tentacle-active {
            display: block !important;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .multi-backend-stats {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
            margin: 25px 0;
        }
        .stat-card {
            background: linear-gradient(45deg, rgba(74, 105, 189, 0.3), rgba(116, 185, 255, 0.2));
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid #4a69bd;
            transition: transform 0.3s;
        }
        .stat-card:hover {
            transform: scale(1.05);
        }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #74b9ff;
            text-shadow: 0 0 10px #4a69bd;
        }
        .chat-area { 
            background: rgba(0,0,0,0.7); 
            border-radius: 15px; 
            padding: 25px; 
            margin: 25px 0; 
            min-height: 450px; 
            max-height: 600px; 
            overflow-y: auto; 
            font-family: 'Courier New', monospace;
            border: 2px solid #4a69bd;
        }
        input, button { 
            padding: 15px; 
            font-size: 1.1em; 
            border: none; 
            border-radius: 10px; 
            margin: 8px; 
        }
        input { 
            background: rgba(255,255,255,0.1); 
            color: white; 
            width: 75%; 
            border: 2px solid #4a69bd;
        }
        button { 
            background: linear-gradient(45deg, #4a69bd, #74b9ff); 
            color: white; 
            cursor: pointer; 
            transition: all 0.3s; 
            border: 2px solid #74b9ff;
            font-weight: bold;
        }
        button:hover { 
            transform: scale(1.05); 
            box-shadow: 0 0 20px rgba(74, 105, 189, 0.5);
        }
        .backend-commands {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }
        .backend-cmd {
            background: linear-gradient(45deg, rgba(0, 184, 148, 0.2), rgba(85, 239, 196, 0.1));
            padding: 15px;
            border-radius: 10px;
            cursor: pointer;
            border: 2px solid #00b894;
            text-align: center;
            font-size: 1em;
            transition: all 0.3s;
        }
        .backend-cmd:hover {
            background: linear-gradient(45deg, rgba(0, 184, 148, 0.4), rgba(85, 239, 196, 0.3));
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 184, 148, 0.3);
        }
        .capabilities {
            background: rgba(0, 184, 148, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #00b894;
        }
    </style>
</head>
<body>
    <div class="backend-indicator">
        🤖 Backend: ${hiveStatus.llmConfig.backend}<br>
        🧠 Model: ${hiveStatus.llmConfig.model}
    </div>
    
    <div class="tentacle-indicator" id="tentacleIndicator">
        🐙 Research Tentacles Deployed<br>
        <span id="tentacleStatus">Searching...</span>
    </div>
    
    <div class="container">
        <h1>Lillith Queen Professional</h1>
        <p style="text-align: center; color: #74b9ff; font-size: 1.3em; text-shadow: 0 0 10px #4a69bd;">
            🌐 Multi-Backend LLM • 🐙 Research Tentacles • 🔧 Comprehensive Troubleshooting • 🚫 No Workarounds
        </p>
        
        <div class="capabilities">
            <h3 style="color: #00b894;">🌐 Multi-Backend LLM Intelligence</h3>
            <p><strong>Auto-Detection:</strong> Finds and connects to any LLM backend:</p>
            <p>• Ollama (11434) - Auto-installs Codestral 3B if needed</p>
            <p>• vLLM (8000) - High-performance production inference</p>
            <p>• LM Studio (1234) - User-friendly GUI interface</p>
            <p>• llama.cpp (8080) - Lightweight C++ implementation</p>
            <p>• text-generation-webui (5000) - Feature-rich web interface</p>
            <p><strong>Research Tentacles:</strong> Extend knowledge for unknown problems in real-time</p>
        </div>
        
        <div class="multi-backend-stats">
            <div class="stat-card">
                <div class="stat-number">${hiveStatus.hiveStats.activeDrones}</div>
                <div>Multi-LLM Drones</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${hiveStatus.llmConfig.conversationHistory}</div>
                <div>LLM Conversations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${hiveStatus.hiveStats.researchSessions}</div>
                <div>Research Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${hiveStatus.llmConfig.backend === 'unknown' ? '0' : '1'}</div>
                <div>Backends Active</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">3B</div>
                <div>Model Size</div>
            </div>
        </div>
        
        <div class="backend-commands">
            <div class="backend-cmd" onclick="sendCommand('Auto-detect what LLM backends I have running')">🔍 Scan LLM Backends</div>
            <div class="backend-cmd" onclick="sendCommand('Research this error I\\'ve never encountered before')">🐙 Deploy Research Tentacles</div>
            <div class="backend-cmd" onclick="sendCommand('My system has performance issues - investigate with full capabilities')">⚡ Full Multi-Backend Analysis</div>
            <div class="backend-cmd" onclick="sendCommand('Switch to the best available LLM backend for troubleshooting')">🔄 Optimize Backend Selection</div>
            <div class="backend-cmd" onclick="sendCommand('Test all my LLM backends and report their capabilities')">🧪 Backend Capability Test</div>
            <div class="backend-cmd" onclick="sendCommand('Learn about this new technology and extend my knowledge base')">🌐 Knowledge Extension</div>
        </div>
        
        <div class="chat-area" id="chatArea">
            <div style="color: #74b9ff; margin-bottom: 15px; text-shadow: 0 0 5px #4a69bd;">
                <strong>LILLITH QUEEN PROFESSIONAL (${hiveStatus.llmConfig.backend}/${hiveStatus.llmConfig.model}):</strong> Hello! I'm your multi-backend LLM troubleshooting specialist.
                
                <div style="margin: 10px 0; padding: 10px; background: rgba(0, 184, 148, 0.1); border-left: 4px solid #00b894; border-radius: 4px;">
                    <strong>🌐 MULTI-BACKEND INTELLIGENCE</strong><br>
                    I automatically detect and connect to any LLM backend you have running.<br>
                    Currently using: <strong>${hiveStatus.llmConfig.backend}/${hiveStatus.llmConfig.model}</strong>
                </div>
                
                <div style="margin: 10px 0; padding: 10px; background: rgba(108, 92, 231, 0.1); border-left: 4px solid #6c5ce7; border-radius: 4px;">
                    <strong>🐙 RESEARCH TENTACLES READY</strong><br>
                    When I encounter unknown problems, I deploy research tentacles to extend my knowledge.
                </div>
                
                Try saying: "My system has a weird issue I've never seen before" or "Research the best solution for this problem"
            </div>
        </div>
        
        <div>
            <input type="text" id="messageInput" placeholder="Describe any problem - I'll use the best LLM backend + research tentacles..." onkeypress="if(event.key==='Enter') sendCommand()">
            <button onclick="sendCommand()">Send Message</button>
            <button onclick="refreshHive()">Refresh Status</button>
            <button onclick="showCapabilities()">Show Capabilities</button>
        </div>
        
        <div style="margin-top: 15px; text-align: center;">
            <button onclick="downloadVerboseReport()" style="background: linear-gradient(45deg, #6c5ce7, #a29bfe);">📄 Download Full Report</button>
            <button onclick="showComprehensiveReport()" style="background: linear-gradient(45deg, #00b894, #55efc4);">📊 View Reports</button>
            <button onclick="showSystemInventory()" style="background: linear-gradient(45deg, #fd79a8, #fdcb6e);">🔍 System Inventory</button>
            <button onclick="toggleVerboseMode()" style="background: linear-gradient(45deg, #e17055, #fab1a0);">🔊 Verbose Mode</button>
        </div>
    </div>

    <script>
        function showTentacleActivity(status) {
            const indicator = document.getElementById('tentacleIndicator');
            const statusSpan = document.getElementById('tentacleStatus');
            indicator.classList.add('tentacle-active');
            statusSpan.textContent = status;
        }
        
        function hideTentacleActivity() {
            const indicator = document.getElementById('tentacleIndicator');
            indicator.classList.remove('tentacle-active');
        }
        
        function sendCommand(command = null) {
            const input = document.getElementById('messageInput');
            const chatArea = document.getElementById('chatArea');
            
            const cmd = command || input.value;
            if (!cmd.trim()) return;
            
            chatArea.innerHTML += '<div style="color: #74b9ff; margin: 12px 0; text-shadow: 0 0 5px #4a69bd;"><strong>💬 YOU:</strong> ' + cmd + '</div>';
            
            const needsResearch = cmd.toLowerCase().includes('research') || 
                                 cmd.toLowerCase().includes('never seen') ||
                                 cmd.toLowerCase().includes('unknown') ||
                                 cmd.toLowerCase().includes('weird') ||
                                 cmd.toLowerCase().includes('investigate');
            
            if (needsResearch) {
                showTentacleActivity('Research tentacles + multi-backend analysis...');
            }
            
            const thinkingId = 'thinking-' + Date.now();
            chatArea.innerHTML += '<div id="' + thinkingId + '" style="color: #a29bfe; margin: 8px 0; font-style: italic;">🌐 Multi-backend LLM analyzing' + (needsResearch ? ' + 🐙 tentacles researching...' : '...') + '</div>';
            chatArea.scrollTop = chatArea.scrollHeight;
            
            fetch('/api/queen/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: cmd })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById(thinkingId).remove();
                hideTentacleActivity();
                
                if (data.type === 'multi_backend_tentacle_response') {
                    chatArea.innerHTML += '<div style="color: #74b9ff; margin: 8px 0; padding: 10px; background: rgba(116, 185, 255, 0.1); border-radius: 8px; border-left: 4px solid #74b9ff;"><strong>🧠 LILLITH (' + (data.backend || 'unknown') + '/' + (data.model || 'unknown') + '):</strong><br>' + data.conversational.replace(/\\n/g, '<br>') + '</div>';
                    
                    if (data.tentaclesDeployed && data.tentaclesDeployed.length > 0) {
                        let researchHtml = '<div style="color: #6c5ce7; margin: 8px 0; padding: 10px; background: rgba(108, 92, 231, 0.1); border-radius: 8px; border-left: 4px solid #6c5ce7;">';
                        researchHtml += '<strong>🐙 RESEARCH TENTACLES DEPLOYED:</strong><br>';
                        data.tentaclesDeployed.forEach(tentacle => {
                            researchHtml += '• ' + tentacle + '<br>';
                        });
                        if (data.research) {
                            researchHtml += '<br><strong>Research Results:</strong><br>' + data.research.replace(/\\n/g, '<br>');
                        }
                        researchHtml += '</div>';
                        chatArea.innerHTML += researchHtml;
                    }
                }
                chatArea.scrollTop = chatArea.scrollHeight;
            })
            .catch(error => {
                document.getElementById(thinkingId).remove();
                hideTentacleActivity();
                chatArea.innerHTML += '<div style="color: #ff6b6b;">❌ Error: ' + error.message + '</div>';
                chatArea.scrollTop = chatArea.scrollHeight;
            });
            
            if (!command) input.value = '';
        }

        function refreshHive() { location.reload(); }
        function showCapabilities() { /* Implementation */ }
        function downloadVerboseReport() { /* Implementation */ }
        function showComprehensiveReport() { /* Implementation */ }
        function showSystemInventory() { /* Implementation */ }
        function toggleVerboseMode() { /* Implementation */ }
        
        document.addEventListener('DOMContentLoaded', function() {
            const chatArea = document.getElementById('chatArea');
            setTimeout(() => {
                chatArea.innerHTML += '<div style="color: #00b894; margin: 10px 0; font-style: italic; padding: 8px; background: rgba(0, 184, 148, 0.1); border-radius: 6px;">💡 <strong>Multi-Backend Ready!</strong> I can work with any LLM backend you have running and extend my knowledge with research tentacles!</div>';
                chatArea.scrollTop = chatArea.scrollHeight;
            }, 2000);
        });
    </script>
</body>
</html>`;
    
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
    return;
  }
  
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('404 Not Found');
});

server.listen(5003, () => {
  console.log('[QUEEN] 🚀 LILLITH QUEEN PROFESSIONAL MULTI-BACKEND + TENTACLES ACTIVE!');
  console.log('[SYSTEM] Access interface at: http://localhost:5003');
  console.log('[BACKENDS] Auto-detecting: Ollama, vLLM, LM Studio, llama.cpp, text-gen-webui');
  console.log('[TENTACLES] Research tentacle system: ARMED FOR KNOWLEDGE EXTENSION');
  console.log('[PHILOSOPHY] No workarounds policy: ENFORCED');
  console.log('[STATUS] Universal troubleshooting system ready - adapts to any environment!');
});