// KVision :: MedMatrix Technical Documentation JavaScript Interactions
document.addEventListener('DOMContentLoaded', () => {

  // --- Real-time Timestamp Ticker ---
  const liveTimestamp = document.getElementById('live-timestamp');
  function updateTimestamp() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const yyyy = now.getFullYear();
    const mm = pad(now.getMonth() + 1);
    const dd = pad(now.getDate());
    const hh = pad(now.getHours());
    const min = pad(now.getMinutes());
    const ss = pad(now.getSeconds());
    liveTimestamp.textContent = `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
  }
  setInterval(updateTimestamp, 1000);
  updateTimestamp();

  // --- Navigation & Scrollspy Section Highlight ---
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('nav.sidebar a');

  function updateActiveSection() {
    let currentSectionId = '';
    
    // Find the section currently in viewport
    sections.forEach(sec => {
      const top = sec.offsetTop;
      if (window.scrollY >= top - 120) {
        currentSectionId = sec.id;
      }
    });

    if (currentSectionId) {
      navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === '#' + currentSectionId) {
          link.classList.add('active');
        } else {
          link.classList.remove('active');
        }
      });
    }
  }

  // Instant scroll-to feedback
  navLinks.forEach(link => {
    link.addEventListener('click', () => {
      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    });
  });

  window.addEventListener('scroll', updateActiveSection, { passive: true });
  updateActiveSection(); // Run on init

  // --- Copy Code to Clipboard ---
  function handleCopy(btn) {
    const codeBlock = btn.closest('.code-block');
    if (!codeBlock) return;
    
    const preEl = codeBlock.querySelector('pre');
    if (!preEl) return;

    // Use textContent to get clean code text (avoid HTML markup)
    const textToCopy = preEl.textContent;

    navigator.clipboard.writeText(textToCopy).then(() => {
      const originalText = btn.textContent;
      btn.textContent = 'copied!';
      btn.classList.add('copied');

      setTimeout(() => {
        btn.textContent = originalText;
        btn.classList.remove('copied');
      }, 1800);
    }).catch(err => {
      console.error('Failed to copy text: ', err);
    });
  }

  // Event Delegation for copy buttons
  document.addEventListener('click', (e) => {
    if (e.target && e.target.classList.contains('copy-btn')) {
      handleCopy(e.target);
    }
  });

  // Global helper in case inline onclick exists
  window.copyCode = function(btn) {
    handleCopy(btn);
  };

  // --- Visual Explorer Directory Tree Interactivity ---
  const folders = document.querySelectorAll('.tree-node.folder');
  const files = document.querySelectorAll('.tree-node.file');
  const detailsCard = document.getElementById('details-card-popover');
  const detailsPath = document.getElementById('details-node-path');
  const detailsBadge = document.getElementById('details-node-badge');
  const detailsDesc = document.getElementById('details-node-desc');

  // Toggle Folder expansion/collapse
  folders.forEach(folder => {
    folder.addEventListener('click', () => {
      const folderId = folder.id;
      const childContainerId = folderId.replace('fold-', 'child-');
      const childContainer = document.getElementById(childContainerId);
      
      if (childContainer) {
        const isCollapsed = folder.classList.toggle('collapsed');
        childContainer.classList.toggle('hidden', isCollapsed);
        folder.querySelector('.tree-expander').textContent = isCollapsed ? '▶' : '▼';
      }
    });
  });

  // Display File details on click
  files.forEach(file => {
    file.addEventListener('click', () => {
      // Remove selected class from others
      files.forEach(f => f.classList.remove('selected'));
      file.classList.add('selected');

      const path = file.getAttribute('data-path');
      const badge = file.getAttribute('data-badge').toUpperCase();
      const desc = file.getAttribute('data-desc');

      detailsPath.textContent = path;
      detailsBadge.textContent = badge;
      detailsDesc.textContent = desc;

      // Update badge style class dynamically
      detailsBadge.className = 'tech-badge';
      if (badge === 'PYTHON' || badge === 'ONNX') detailsBadge.classList.add('badge-py');
      else if (badge === 'TYPESCRIPT' || badge === 'TSX / REACT' || badge === 'VANILLA CSS') detailsBadge.classList.add('badge-ts');
      else if (badge === 'RUST SOURCE' || badge === 'RUST CARGO') detailsBadge.classList.add('badge-rs');
      else if (badge === 'DOCKER') detailsBadge.classList.add('badge-docker');
      else if (badge === 'PRISMA') detailsBadge.classList.add('badge-db');
      else detailsBadge.classList.add('badge-config');

      detailsCard.classList.add('active');
    });
  });

  // --- Sidebar Search & Explorer Tree Filtering ---
  const searchInput = document.getElementById('doc-search');
  searchInput.addEventListener('input', () => {
    const query = searchInput.value.toLowerCase().trim();
    
    // 1. Filter sidebar nav section headers and links
    navLinks.forEach(link => {
      const text = link.textContent.toLowerCase();
      const sectionId = link.getAttribute('href').replace('#', '');
      const section = document.getElementById(sectionId);
      
      let matches = text.includes(query);
      if (!matches && section) {
        matches = section.textContent.toLowerCase().includes(query);
      }

      if (matches || query === '') {
        link.style.display = 'block';
      } else {
        link.style.display = 'none';
      }
    });

    // 2. Filter interactive folder tree nodes
    if (query === '') {
      // Restore default view: remove hidden/visible classes
      files.forEach(file => file.style.display = 'flex');
      folders.forEach(folder => {
        folder.style.display = 'flex';
        folder.classList.remove('collapsed');
        folder.querySelector('.tree-expander').textContent = '▼';
        const childId = folder.id.replace('fold-', 'child-');
        const child = document.getElementById(childId);
        if (child) child.classList.remove('hidden');
      });
    } else {
      // Temporary list of matching node parents
      const matchParents = new Set();

      files.forEach(file => {
        const fileName = file.querySelector('.tree-label').textContent.toLowerCase();
        const fileDesc = (file.getAttribute('data-desc') || '').toLowerCase();
        const filePath = (file.getAttribute('data-path') || '').toLowerCase();
        const matches = fileName.includes(query) || fileDesc.includes(query) || filePath.includes(query);

        if (matches) {
          file.style.display = 'flex';
          // Save folder hierarchy parents
          let parentNode = file.parentElement;
          while (parentNode && parentNode.id !== 'visual-explorer-tree') {
            if (parentNode.classList.contains('tree-children')) {
              const foldId = parentNode.id.replace('child-', 'fold-');
              matchParents.add(foldId);
            }
            parentNode = parentNode.parentElement;
          }
        } else {
          file.style.display = 'none';
        }
      });

      // Filter folder nodes based on matching children hierarchy
      folders.forEach(folder => {
        const folderName = folder.querySelector('.tree-label').textContent.toLowerCase();
        const isParentOfMatch = matchParents.has(folder.id);
        const matchesSelf = folderName.includes(query);

        if (isParentOfMatch || matchesSelf) {
          folder.style.display = 'flex';
          folder.classList.remove('collapsed'); // expand matching structure
          folder.querySelector('.tree-expander').textContent = '▼';
          const childId = folder.id.replace('fold-', 'child-');
          const child = document.getElementById(childId);
          if (child) child.classList.remove('hidden');
        } else {
          folder.style.display = 'none';
          const childId = folder.id.replace('fold-', 'child-');
          const child = document.getElementById(childId);
          if (child) child.classList.add('hidden');
        }
      });
    }
  });

  // --- Interactive MRI Pipeline Simulator Logic ---
  const simContrast = document.getElementById('sim-contrast');
  const simNoise = document.getElementById('sim-noise');
  const simMotion = document.getElementById('sim-motion');
  const simWrap = document.getElementById('sim-wrap');
  const simThreshold = document.getElementById('sim-threshold');
  const runSimBtn = document.getElementById('run-simulation-btn');

  // Sliders value feedback
  const sliders = [
    { el: simNoise, val: 'val-noise' },
    { el: simMotion, val: 'val-motion' },
    { el: simWrap, val: 'val-wrap' },
    { el: simThreshold, val: 'val-threshold' }
  ];

  sliders.forEach(slider => {
    slider.el.addEventListener('input', () => {
      document.getElementById(slider.val).textContent = parseFloat(slider.el.value).toFixed(2);
    });
  });

  // Console log outputs
  const consoleOutput = document.getElementById('sim-console-output');
  function logToConsole(message, type = 'info') {
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    const timestamp = new Date().toLocaleTimeString();
    line.textContent = `[${timestamp}] ${message}`;
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
  }

  // Flowchart Nodes control
  const nodeIngest = document.getElementById('node-ingest');
  const nodeGating = document.getElementById('node-gating');
  const nodeRecon = document.getElementById('node-recon');
  const nodeClassify = document.getElementById('node-classify');

  function resetFlowNodes() {
    const nodes = [nodeIngest, nodeGating, nodeRecon, nodeClassify];
    nodes.forEach(node => {
      node.className = 'pipe-node-status';
      node.querySelector('.status-indicator-dot').style.background = '#94a3b8';
    });
  }

  // Pathology Classes list (matching kvision_inference.h labels)
  const PATHOLOGY_CLASSES = [
    'Normal',
    'Tumor_Glioma',
    'Ischemia',
    'MS_Lesions',
    'Hydrocephalus',
    'Atrophy',
    'Hemorrhage',
    'Cerebral_Cyst',
    'Edema',
    'AVM',
    'Cerebral_Microbleeds'
  ];

  // Initialize empty chart bars
  const chartBarsContainer = document.getElementById('chart-bars');
  function initChart() {
    chartBarsContainer.innerHTML = '';
    PATHOLOGY_CLASSES.forEach((pName, index) => {
      const row = document.createElement('div');
      row.className = 'chart-bar-row';
      row.innerHTML = `
        <div class="chart-bar-label">${index}. ${pName}</div>
        <div class="chart-bar-outer">
          <div class="chart-bar-inner" id="bar-${index}"></div>
        </div>
        <div class="chart-bar-val" id="val-${index}">0.0%</div>
      `;
      chartBarsContainer.appendChild(row);
    });
  }
  initChart();

  // Run Simulation Event
  runSimBtn.addEventListener('click', () => {
    // Disable button during execution
    runSimBtn.disabled = true;
    runSimBtn.querySelector('span').textContent = 'PROCESSING SIMULATION...';
    
    resetFlowNodes();
    initChart();
    
    // Read input values
    const contrastVal = parseInt(simContrast.value);
    const noiseVal = parseFloat(simNoise.value);
    const motionVal = parseFloat(simMotion.value);
    const wrapVal = parseFloat(simWrap.value);
    const thresholdVal = parseFloat(simThreshold.value);

    // Calculate composite scores
    const compositeScore = Math.max(noiseVal, motionVal, wrapVal);
    const isGatedOut = compositeScore >= thresholdVal;

    logToConsole('--- NEW MRI PROCESSING RUN INITIATED ---', 'info');
    
    // Step 1: Ingest
    setTimeout(() => {
      nodeIngest.classList.add('success');
      nodeIngest.querySelector('.status-indicator-dot').style.background = '#22c55e';
      logToConsole(`[INGEST] Uploaded study raw complex arrays. Shape: [1, 8, 16, 128, 128, 2]`, 'success');
      logToConsole(`[INGEST] Saved raw file: minio://kspace-raw/study-uuid-raw.npy`, 'info');
      
      // Step 2: Gating
      setTimeout(() => {
        nodeGating.classList.add('active');
        logToConsole(`[TIER 1] SSM Estimator evaluating K-Space channel matrix...`, 'info');
        
        setTimeout(() => {
          logToConsole(`[TIER 1] Channel metrics - Noise: ${noiseVal.toFixed(2)}, Motion: ${motionVal.toFixed(2)}, Phase: ${wrapVal.toFixed(2)}`, 'info');
          logToConsole(`[TIER 1] Calculated Composite Artifact Score: ${compositeScore.toFixed(2)} (Threshold Limit: ${thresholdVal.toFixed(2)})`, 'info');
          
          if (isGatedOut) {
            nodeGating.classList.add('failed');
            nodeGating.querySelector('.status-indicator-dot').style.background = '#ef4444';
            nodeRecon.classList.add('gated-out');
            nodeClassify.classList.add('gated-out');
            
            logToConsole(`[TIER 1] WARNING: Artifact level exceeded threshold. Gating triggered.`, 'warn');
            logToConsole(`[BACKEND] BullMQ job study-uuid state set to [GATED_SKIP]`, 'warn');
            logToConsole(`[DATABASE] Saved GatingDecision: imageEncoderTriggered = false, reason = "Severe artifacts (Score: ${compositeScore.toFixed(2)})"`, 'success');
            logToConsole(`[DATABASE] Saved ModelResult & AnomalyDetection records.`, 'success');
            logToConsole(`[SYSTEM] Finished processing study in gated mode. Reconstructions bypassed to save compute.`, 'success');
            
            // Populate gated probabilities (0% for all, normal 0%)
            fillBars(null);
            
            // Re-enable button
            runSimBtn.disabled = false;
            runSimBtn.querySelector('span').textContent = 'RUN CLINICAL PIPELINE';
          } else {
            nodeGating.classList.add('success');
            nodeGating.querySelector('.status-indicator-dot').style.background = '#22c55e';
            logToConsole(`[TIER 1] Clean acquisition verification success. Processing image encoders.`, 'success');
            
            // Step 3: Recon, Registration & Denoise
            setTimeout(() => {
              nodeRecon.classList.add('active');
              logToConsole(`[TIER 2] Computing 2D Centered IFFT & RSS Coil combination...`, 'info');
              
              setTimeout(() => {
                logToConsole(`[TIER 2] Running SimpleITK rigid registration (Mean Squares metric)...`, 'info');
                logToConsole(`[TIER 2] Executing PyTorch DnCNN residual denoiser model...`, 'info');
                
                setTimeout(() => {
                  nodeRecon.classList.add('success');
                  nodeRecon.querySelector('.status-indicator-dot').style.background = '#22c55e';
                  logToConsole(`[TIER 2] Reconstruction finished. Reconstructed magnitude matrix saved to MinIO.`, 'success');
                  logToConsole(`[TIER 2] Saved file: minio://reconstructed-magnitude/study-uuid-recon.npy`, 'info');
                  
                  // Step 4: Classification
                  setTimeout(() => {
                    nodeClassify.classList.add('active');
                    logToConsole(`[TIER 3] Loading Fused S4-CNN Volumetric model weights into CUDA device memory...`, 'info');
                    
                    setTimeout(() => {
                      logToConsole(`[TIER 3] Running cross-attention spatial-frequency sequence classification...`, 'info');
                      
                      setTimeout(() => {
                        nodeClassify.classList.add('success');
                        nodeClassify.querySelector('.status-indicator-dot').style.background = '#22c55e';
                        
                        // Compute probabilities
                        const probs = generateProbabilities(contrastVal, noiseVal, motionVal);
                        fillBars(probs);
                        
                        const maxIndex = probs.indexOf(Math.max(...probs));
                        const predName = PATHOLOGY_CLASSES[maxIndex];
                        const confPercent = (probs[maxIndex] * 100).toFixed(1);
                        
                        logToConsole(`[TIER 3] Inference completed in 11.8 ms (CUDA EP).`, 'success');
                        logToConsole(`[TIER 3] PREDICTED PATHOLOGY: ${predName} (${confPercent}% confidence)`, 'success');
                        logToConsole(`[DATABASE] Saved ModelResult: reconstructedKey = "study-uuid-recon.npy", confidenceScore = ${(probs[maxIndex]).toFixed(4)}`, 'success');
                        logToConsole(`[DATABASE] Saved GatingDecision: imageEncoderTriggered = true`, 'success');
                        logToConsole(`[SYSTEM] Full study processing successfully completed. Report status marked complete.`, 'success');
                        
                        // Re-enable button
                        runSimBtn.disabled = false;
                        runSimBtn.querySelector('span').textContent = 'RUN CLINICAL PIPELINE';
                      }, 800);
                    }, 500);
                  }, 500);
                }, 600);
              }, 600);
            }, 500);
          }
        }, 800);
      }, 500);
    }, 500);
  });

  // Dynamic Probabilities Generator based on sliders
  function generateProbabilities(contrast, noise, motion) {
    let scores = new Array(11).fill(0);
    
    // Add baseline values
    if (contrast === 0) { // T1-weighted
      // T1 shows tumor structures and normal anatomy well
      scores[0] = 0.40; // Normal
      scores[1] = 0.55; // Tumor Glioma
      scores[7] = 0.35; // Cerebral Cyst
      scores[6] = 0.20; // Hemorrhage
      scores[10] = 0.15; // Cerebral Microbleeds
    } else { // T2-weighted
      // T2 shows fluids/edema/lesions extremely bright
      scores[0] = 0.20; // Normal
      scores[8] = 0.65; // Edema
      scores[3] = 0.50; // MS Lesions
      scores[4] = 0.45; // Hydrocephalus
      scores[2] = 0.25; // Ischemia
    }

    // Add noise/motion impact (causes anomalies/distortion, shifts probabilities)
    scores[0] -= (noise * 0.15 + motion * 0.15); // Less likely to be classified as completely Normal
    scores[5] += (motion * 0.3); // Motion artifacts might resemble Atrophy due to blurring
    scores[8] += (noise * 0.2);  // Noise highlights look like fluid/edema high intensity

    // Add minor random variation
    scores = scores.map(s => Math.max(0.02, s + Math.random() * 0.1));

    // Normalize using Softmax (exponential scale to make one class stand out)
    const exps = scores.map(s => Math.exp(s * 8)); // scale by 8 to accentuate difference
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumExps);
    
    return probs;
  }

  // Animate Bar chart filling
  function fillBars(probs) {
    PATHOLOGY_CLASSES.forEach((pName, index) => {
      const bar = document.getElementById(`bar-${index}`);
      const val = document.getElementById(`val-${index}`);
      
      if (probs === null) {
        // Gated out or cleared
        bar.style.width = '0%';
        bar.classList.remove('high');
        val.textContent = '0.0%';
      } else {
        const pVal = probs[index];
        const percent = (pVal * 100).toFixed(1);
        bar.style.width = `${percent}%`;
        
        // Mark high alert if not Normal and probability is high
        if (index !== 0 && pVal > 0.4) {
          bar.classList.add('high');
        } else {
          bar.classList.remove('high');
        }
        
        val.textContent = `${percent}%`;
      }
    });
  }

});
