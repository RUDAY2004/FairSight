class FairSightDashboard {
  constructor() {
    this.currentPage = "overview";
    this.isDarkMode = localStorage.getItem("darkMode") === "true";
    this.uploadedData = {};
    this.analysisResults = {};

    this.init();
  }

  init() {
    this.setupEventListeners();
    this.setupDarkMode();
    this.setupNavigation();
    this.setupFileUploads();
    this.setupCharts();
    this.loadSampleData();
  }

  setupEventListeners() {
    // Dark mode toggle
    document.getElementById("darkModeToggle").addEventListener("click", () => {
      this.toggleDarkMode();
    });

    // Profile dropdown
    document.getElementById("profileBtn").addEventListener("click", (e) => {
      e.stopPropagation();
      this.toggleProfileDropdown();
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", () => {
      this.closeProfileDropdown();
    });

    // Navigation
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        const page = item.dataset.page;
        this.navigateToPage(page);
      });
    });

    // Industry cards and buttons navigation
    document.querySelectorAll(".industry-card, .industry-card .btn").forEach((element) => {
      element.addEventListener("click", (e) => {
        e.stopPropagation();
        const page = element.dataset.page;
        if (page) {
          this.navigateToPage(page);
        }
      });
    });

    // Settings
    document.getElementById("themeSelect")?.addEventListener("change", (e) => {
      this.handleThemeChange(e.target.value);
    });

    // Responsive sidebar toggle
    this.setupResponsiveSidebar();

    // Analysis buttons
    document.getElementById("hrAnalyze")?.addEventListener("click", () => this.analyzeIndustry("hr"));
    document.getElementById("bankingAnalyze")?.addEventListener("click", () => this.analyzeIndustry("banking"));
    document.getElementById("healthcareAnalyze")?.addEventListener("click", () => this.analyzeIndustry("healthcare"));
    document.getElementById("retailAnalyze")?.addEventListener("click", () => this.analyzeIndustry("retail"));
    document.getElementById("educationAnalyze")?.addEventListener("click", () => this.analyzeIndustry("education"));
  }

  setupDarkMode() {
    if (this.isDarkMode) {
      document.documentElement.setAttribute("data-theme", "dark");
    }
  }

  toggleDarkMode() {
    this.isDarkMode = !this.isDarkMode;
    localStorage.setItem("darkMode", this.isDarkMode);
    if (this.isDarkMode) {
      document.documentElement.setAttribute("data-theme", "dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }

  toggleProfileDropdown() {
    const dropdown = document.getElementById("profileDropdown");
    dropdown.classList.toggle("show");
  }

  closeProfileDropdown() {
    const dropdown = document.getElementById("profileDropdown");
    dropdown.classList.remove("show");
  }

  setupNavigation() {
    this.updateActiveNavItem("overview");
  }

  navigateToPage(page) {
    document.querySelectorAll(".page").forEach((p) => {
      p.classList.remove("active");
    });
    const targetPage = document.getElementById(`${page}-page`);
    if (targetPage) {
      targetPage.classList.add("active");
      this.currentPage = page;
      this.updateActiveNavItem(page);
    }
  }

  updateActiveNavItem(page) {
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.classList.remove("active");
    });
    const activeItem = document.querySelector(`[data-page="${page}"]`);
    if (activeItem) {
      activeItem.classList.add("active");
    }
  }

  setupFileUploads() {
    // Industry-specific uploads
    const industries = ["hr", "banking", "healthcare", "retail", "education"];
    industries.forEach((industry) => {
      this.setupFileUpload(`${industry}File`, `${industry}Upload`, (file, data) => {
        this.handleIndustryUpload(industry, file, data);
      });
    });

    // General upload center
    this.setupFileUpload("uploadFile", null, (file, data) => {
      this.handleGeneralUpload(file, data);
    });

    // Run analysis button
    document.getElementById("runAnalysis")?.addEventListener("click", () => {
      this.runGeneralAnalysis();
    });

    // Generate certificate button
    document.getElementById("generateCertificate")?.addEventListener("click", () => {
      this.generateCertificate();
    });
  }

  setupFileUpload(inputId, uploadAreaId, callback) {
    const input = document.getElementById(inputId);
    const uploadArea = uploadAreaId ? document.getElementById(uploadAreaId) : null;

    if (!input) return;

    input.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        this.processFile(file, callback);
      }
    });

    if (uploadArea) {
      uploadArea.addEventListener("click", () => {
        input.click();
      });

      uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = "var(--primary-color)";
      });

      uploadArea.addEventListener("dragleave", () => {
        uploadArea.style.borderColor = "var(--border-color)";
      });

      uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = "var(--border-color)";
        const file = e.dataTransfer.files[0];
        if (file) {
          this.processFile(file, callback);
        }
      });
    }
  }

  processFile(file, callback) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        let data;
        if (file.name.endsWith(".csv")) {
          data = this.parseCSV(e.target.result);
        } else {
          throw new Error("Unsupported file format");
        }
        callback(file, data);
      } catch (error) {
        console.error("Error processing file:", error);
        this.showNotification("Error processing file", "error");
      }
    };
    reader.readAsText(file);
  }

  parseCSV(csvText) {
    const lines = csvText.split("\n");
    const headers = lines[0].split(",").map((h) => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
      if (lines[i].trim()) {
        const values = lines[i].split(",").map((v) => v.trim());
        const row = {};
        headers.forEach((header, index) => {
          row[header] = values[index] || "";
        });
        data.push(row);
      }
    }

    return { headers, data };
  }

  handleIndustryUpload(industry, file, data) {
    this.uploadedData[industry] = { file, data };
    document.getElementById(`${industry}Results`).style.display = "block";
    this.showNotification(`File uploaded for ${industry} analysis`, "success");
  }

  analyzeIndustry(industry) {
    const data = this.uploadedData[industry]?.data;
    if (!data) {
      this.showNotification("Please upload a CSV file first", "error");
      return;
    }

    this.showNotification(`Analyzing ${industry} data...`, "info");
    setTimeout(() => {
      this.showIndustryResults(industry, data);
      this.showNotification("Analysis completed successfully!", "success");
    }, 2000);
  }

  showIndustryResults(industry, data) {
    const biasResultsDiv = document.getElementById(`${industry}BiasResults`);
    const fairnessMetricsDiv = document.getElementById(`${industry}FairnessMetrics`);

    // Simulate bias analysis
    const biasTypes = {
      hr: ["Gender Bias", "Age Bias", "Ethnicity Bias"],
      banking: ["Income Bias", "Credit Score Bias", "Geographic Bias"],
      healthcare: ["Diagnosis Bias", "Demographic Bias", "Treatment Bias"],
      retail: ["Preference Bias", "Purchase History Bias", "Demographic Bias"],
      education: ["Grade Bias", "Socioeconomic Bias", "Performance Bias"],
    };

    const biasResults = biasTypes[industry].map((bias) => ({
      type: bias,
      severity: ["Low", "Medium", "High"][Math.floor(Math.random() * 3)],
      confidence: (Math.random() * 0.4 + 0.6).toFixed(2),
    }));

    let biasHTML = `
      <table>
        <thead>
          <tr>
            <th>Bias Type</th>
            <th>Severity</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
    `;
    biasResults.forEach((result) => {
      biasHTML += `
        <tr>
          <td>${result.type}</td>
          <td><span class="severity-${result.severity.toLowerCase()}">${result.severity}</span></td>
          <td>${(result.confidence * 100).toFixed(0)}%</td>
        </tr>
      `;
    });
    biasHTML += "</tbody></table>";
    biasResultsDiv.innerHTML = biasHTML;

    // Simulate fairness metrics
    const fairnessMetrics = {
      fairnessScore: Math.floor(Math.random() * 20 + 80),
      transparencyScore: Math.floor(Math.random() * 20 + 80),
      biasMitigationScore: Math.floor(Math.random() * 20 + 80),
    };

    fairnessMetricsDiv.innerHTML = `
      <div class="metrics-list">
        <div class="metric-item">
          <span class="metric-label">Fairness Score</span>
          <span class="metric-value">${fairnessMetrics.fairnessScore}%</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Transparency</span>
          <span class="metric-value">${fairnessMetrics.transparencyScore}%</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Bias Mitigation</span>
          <span class="metric-value">${fairnessMetrics.biasMitigationScore}%</span>
        </div>
      </div>
    `;
  }

  handleGeneralUpload(file, data) {
    this.uploadedData.general = { file, data };
    this.updateUploadHistory(file);
  }

  updateUploadHistory(file) {
    const historyDiv = document.getElementById("uploadHistory");
    const existingTable = historyDiv.querySelector("table");

    if (!existingTable) {
      historyDiv.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Size</th>
              <th>Upload Date</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="historyTableBody">
          </tbody>
        </table>
      `;
    }

    const tbody = document.getElementById("historyTableBody");
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${file.name}</td>
      <td>${file.type || "Unknown"}</td>
      <td>${this.formatFileSize(file.size)}</td>
      <td>${new Date().toLocaleString()}</td>
      <td><span class="status-uploaded">Uploaded</span></td>
    `;
    tbody.insertBefore(row, tbody.firstChild);
  }

  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  runGeneralAnalysis() {
    const modelName = document.getElementById("modelName").value;
    const modelDescription = document.getElementById("modelDescription").value;
    const modelPurpose = document.getElementById("modelPurpose").value;

    if (!modelName || !this.uploadedData.general) {
      this.showNotification("Please provide model name and upload a file", "error");
      return;
    }

    this.showNotification("Running analysis...", "info");
    setTimeout(() => {
      this.showNotification("Analysis completed successfully!", "success");
      this.navigateToPage("scorecard");
    }, 3000);
  }

  setupCharts() {
    this.setupRadarChart();
    this.setupComplianceGauge();
  }

  setupRadarChart() {
    const canvas = document.querySelector("#radarChart canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 100;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const data = [0.9, 0.8, 0.95, 0.75, 0.85];
    const labels = ["Fairness", "Transparency", "Privacy", "Accountability", "Robustness"];

    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--border-color");
    ctx.lineWidth = 1;

    for (let i = 1; i <= 5; i++) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, (radius * i) / 5, 0, 2 * Math.PI);
      ctx.stroke();
    }

    for (let i = 0; i < 5; i++) {
      const angle = (i * 2 * Math.PI) / 5 - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();

      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--text-primary");
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 20);
      ctx.fillText(labels[i], labelX, labelY);
    }

    ctx.beginPath();
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--primary-color") + "40";
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--primary-color");
    ctx.lineWidth = 2;

    for (let i = 0; i < data.length; i++) {
      const angle = (i * 2 * Math.PI) / data.length - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius * data[i];
      const y = centerY + Math.sin(angle) * radius * data[i];

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--primary-color");
    for (let i = 0; i < data.length; i++) {
      const angle = (i * 2 * Math.PI) / data.length - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius * data[i];
      const y = centerY + Math.sin(angle) * radius * data[i];

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  setupComplianceGauge() {
    const gauge = document.getElementById("gaugeProgress");
    if (!gauge) return;

    const score = 85;
    const circumference = 251.2;
    const offset = circumference - (score / 100) * circumference;

    gauge.style.strokeDashoffset = offset;
  }

  generateCertificate() {
    const certificateData = {
      organization: "Your Organization",
      date: new Date().toLocaleDateString(),
      score: 85,
      validUntil: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toLocaleDateString(),
    };

    this.showNotification("Generating ethical certificate...", "info");
    setTimeout(() => {
      this.downloadCertificate(certificateData);
      this.showNotification("Certificate generated successfully!", "success");
    }, 2000);
  }

  downloadCertificate(data) {
    const certificateHTML = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>FairSight AI Ethics Certificate</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
          .certificate { border: 5px solid #0d9488; padding: 50px; max-width: 600px; margin: 0 auto; }
          .title { font-size: 2em; color: #0d9488; margin-bottom: 20px; }
          .score { font-size: 3em; color: #10b981; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="certificate">
          <h1 class="title">AI Ethics Certificate</h1>
          <p>This certifies that</p>
          <h2>${data.organization}</h2>
          <p>has achieved an AI Ethics Compliance Score of</p>
          <div class="score">${data.score}%</div>
          <p>Issued on: ${data.date}</p>
          <p>Valid until: ${data.validUntil}</p>
          <p><em>Powered by FairSight AI Governance Platform</em></p>
        </div>
      </body>
      </html>
    `;

    const blob = new Blob([certificateHTML], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fairsight-certificate-${new Date().toISOString().split("T")[0]}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  setupResponsiveSidebar() {
    if (window.innerWidth <= 1024) {
      const navbar = document.querySelector(".navbar");
      const menuButton = document.createElement("button");
      menuButton.innerHTML = "☰";
      menuButton.className = "mobile-menu-btn";
      menuButton.style.cssText = `
        background: none;
        border: none;
        font-size: 1.5rem;
        color: var(--text-primary);
        cursor: pointer;
        display: block;
      `;
      navbar.querySelector(".nav-left").appendChild(menuButton);

      menuButton.addEventListener("click", () => {
        document.getElementById("sidebar").classList.toggle("open");
      });
    }

    window.addEventListener("resize", () => {
      if (window.innerWidth > 1024) {
        document.getElementById("sidebar").classList.remove("open");
      }
    });
  }

  handleThemeChange(theme) {
    switch (theme) {
      case "dark":
        document.documentElement.setAttribute("data-theme", "dark");
        localStorage.setItem("darkMode", "true");
        break;
      case "light":
        document.documentElement.removeAttribute("data-theme");
        localStorage.setItem("darkMode", "false");
        break;
      case "auto":
        const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        if (prefersDark) {
          document.documentElement.setAttribute("data-theme", "dark");
        } else {
          document.documentElement.removeAttribute("data-theme");
        }
        localStorage.setItem("darkMode", prefersDark.toString());
        break;
    }
  }

  loadSampleData() {
    const sampleUploadHistory = [
      { name: "customer_data.csv", type: "text/csv", size: 1024000, date: "2025-01-14" },
      { name: "model_outputs.json", type: "application/json", size: 512000, date: "2025-01-13" },
      { name: "training_logs.txt", type: "text/plain", size: 256000, date: "2025-01-12" },
    ];

    const historyDiv = document.getElementById("uploadHistory");
    if (historyDiv && !historyDiv.querySelector("table")) {
      let tableHTML = `
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Size</th>
              <th>Upload Date</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
      `;
      sampleUploadHistory.forEach((file) => {
        tableHTML += `
          <tr>
            <td>${file.name}</td>
            <td>${file.type}</td>
            <td>${this.formatFileSize(file.size)}</td>
            <td>${file.date}</td>
            <td><span class="status-processed">Processed</span></td>
          </tr>
        `;
      });
      tableHTML += "</tbody></table>";
      historyDiv.innerHTML = tableHTML;
    }
  }

  showNotification(message, type = "info") {
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 1rem 1.5rem;
      border-radius: 0.5rem;
      color: white;
      font-weight: 500;
      z-index: 10000;
      transform: translateX(100%);
      transition: transform 0.3s ease;
    `;

    switch (type) {
      case "success":
        notification.style.backgroundColor = "var(--success-color)";
        break;
      case "error":
        notification.style.backgroundColor = "var(--error-color)";
        break;
      case "warning":
        notification.style.backgroundColor = "var(--warning-color)";
        break;
      default:
        notification.style.backgroundColor = "var(--primary-color)";
    }

    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
      notification.style.transform = "translateX(0)";
    }, 100);

    setTimeout(() => {
      notification.style.transform = "translateX(100%)";
      setTimeout(() => {
        document.body.removeChild(notification);
      }, 300);
    }, 3000);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.dashboard = new FairSightDashboard();
});

const additionalStyles = `
  .status-uploaded {
    color: var(--primary-color);
    font-weight: 500;
  }
  
  .status-processed {
    color: var(--success-color);
    font-weight: 500;
  }
  
  .severity-high {
    color: var(--error-color);
    font-weight: 500;
  }
  
  .severity-medium {
    color: var(--warning-color);
    font-weight: 500;
  }
  
  .severity-low {
    color: var(--success-color);
    font-weight: 500;
  }
  
  .mobile-menu-btn {
    display: none;
  }
  
  @media (max-width: 1024px) {
    .mobile-menu-btn {
      display: block !important;
    }
    
    .sidebar {
      transform: translateX(-100%);
      transition: transform 0.3s ease;
    }
    
    .sidebar.open {
      transform: translateX(0);
    }
  }
  
  .notification {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
`;

const styleSheet = document.createElement("style");
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);