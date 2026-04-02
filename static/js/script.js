/**
 * ============================================================
 * Plant Disease Detection - Frontend JavaScript
 * ============================================================
 * Features:
 *   - Drag & drop image upload
 *   - Webcam live capture
 *   - API integration with /predict
 *   - Result display with animations
 *   - Dark mode toggle
 *   - Multi-language (EN/Hindi)
 *   - Share functionality
 * ============================================================
 */

'use strict';

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────
const State = {
  selectedFile  : null,
  webcamStream  : null,
  currentLang   : localStorage.getItem('lang') || 'en',
  currentTheme  : localStorage.getItem('theme') || 'light',
  lastResult    : null,
};

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
const API_URL = '/predict';

const TRANSLATIONS = {
  en: {
    analyzing   : 'Analyzing your plant...',
    aiExamining : 'Our AI is examining the leaf patterns',
    uploadFirst : 'Please upload an image first.',
    networkError: 'Network error. Please check your connection.',
    shareTitle  : 'Plant Disease Detection Result',
    shareText   : (name, conf) => `Disease: ${name}\nConfidence: ${(conf * 100).toFixed(1)}%\n\nDetected using PlantDoc AI`,
    copied      : '✓ Link Copied!',
    share       : 'Share Result',
  },
  hi: {
    analyzing   : 'आपके पौधे का विश्लेषण हो रहा है...',
    aiExamining : 'हमारा AI पत्ते के पैटर्न की जांच कर रहा है',
    uploadFirst : 'कृपया पहले एक छवि अपलोड करें।',
    networkError: 'नेटवर्क त्रुटि। कृपया अपना कनेक्शन जांचें।',
    shareTitle  : 'पौधे की बीमारी जांच परिणाम',
    shareText   : (name, conf) => `बीमारी: ${name}\nविश्वास: ${(conf * 100).toFixed(1)}%\n\nPlantDoc AI द्वारा पहचान`,
    copied      : '✓ लिंक कॉपी हो गया!',
    share       : 'परिणाम साझा करें',
  }
};

// ─────────────────────────────────────────────
// INITIALIZATION
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(State.currentTheme);
  applyLanguage(State.currentLang);
  initAnimations();
});

// ─────────────────────────────────────────────
// THEME MANAGEMENT
// ─────────────────────────────────────────────
function toggleTheme() {
  const newTheme = State.currentTheme === 'light' ? 'dark' : 'light';
  applyTheme(newTheme);
}

function applyTheme(theme) {
  State.currentTheme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);

  const icon = document.getElementById('theme-icon');
  if (icon) {
    icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
  }
}

// ─────────────────────────────────────────────
// LANGUAGE MANAGEMENT
// ─────────────────────────────────────────────
function setLanguage(lang) {
  State.currentLang = lang;
  localStorage.setItem('lang', lang);
  applyLanguage(lang);

  // Update active button
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });
}

function applyLanguage(lang) {
  // Update all elements with data-en / data-hi attributes
  document.querySelectorAll(`[data-${lang}]`).forEach(el => {
    const text = el.getAttribute(`data-${lang}`);
    if (text) el.innerHTML = text;
  });

  // Update dynamic result text if available
  if (State.lastResult) {
    document.getElementById('descriptionText').textContent = State.lastResult[`description_${lang}`] || State.lastResult.description || 'No description available.';
    document.getElementById('treatmentText').textContent   = State.lastResult[`treatment_${lang}`] || State.lastResult.treatment || 'No treatment information available.';
    document.getElementById('preventionText').textContent  = State.lastResult[`prevention_${lang}`] || State.lastResult.prevention || 'No prevention information available.';
  }
}

function t(key, ...args) {
  const fn = TRANSLATIONS[State.currentLang][key];
  return typeof fn === 'function' ? fn(...args) : fn;
}

// ─────────────────────────────────────────────
// DRAG & DROP
// ─────────────────────────────────────────────
function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById('dropZone').classList.add('dragover');
}

function handleDragLeave(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('dragover');
}

function handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById('dropZone').classList.remove('dragover');

  const files = e.dataTransfer.files;
  if (files && files.length > 0) {
    processFile(files[0]);
  }
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) processFile(file);
}

// ─────────────────────────────────────────────
// FILE PROCESSING
// ─────────────────────────────────────────────
function processFile(file) {
  // Validate type
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  if (!allowedTypes.includes(file.type)) {
    showNotification('Please upload a valid image file (JPG, PNG, WEBP, BMP)', 'error');
    return;
  }

  // Validate size (16MB)
  if (file.size > 16 * 1024 * 1024) {
    showNotification('File too large. Maximum size is 16MB.', 'error');
    return;
  }

  State.selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    showPreview(e.target.result);
  };
  reader.readAsDataURL(file);

  // Enable analyze button
  enableAnalyzeButton();
}

function showPreview(src) {
  const dropContent     = document.getElementById('dropContent');
  const previewContainer= document.getElementById('previewContainer');
  const previewImage    = document.getElementById('previewImage');

  dropContent.style.display     = 'none';
  previewContainer.style.display= 'block';
  previewImage.src              = src;
}

function resetUpload() {
  State.selectedFile = null;

  document.getElementById('dropContent').style.display      = 'flex';
  document.getElementById('previewContainer').style.display = 'none';
  document.getElementById('fileInput').value                = '';

  // Reset image styles
  const previewImage = document.getElementById('previewImage');
  previewImage.src = '';

  disableAnalyzeButton();
}

function resetAll() {
  resetUpload();
  hideResults();
  stopWebcam();
}

// ─────────────────────────────────────────────
// WEBCAM
// ─────────────────────────────────────────────
async function toggleWebcam() {
  const webcamContainer = document.getElementById('webcamContainer');
  const isVisible = webcamContainer.style.display !== 'none';

  if (isVisible) {
    stopWebcam();
  } else {
    await startWebcam();
  }
}

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }
    });

    State.webcamStream = stream;
    const video = document.getElementById('webcamVideo');
    video.srcObject = stream;

    document.getElementById('webcamContainer').style.display = 'block';
    document.getElementById('webcamBtn').innerHTML =
      '<i class="fas fa-times"></i><span>Stop Camera</span>';

    showNotification('Webcam started! Point at a plant leaf and capture.', 'info');
  } catch (err) {
    console.error('Webcam error:', err);
    showNotification('Could not access webcam. Please check permissions.', 'error');
  }
}

function stopWebcam() {
  if (State.webcamStream) {
    State.webcamStream.getTracks().forEach(track => track.stop());
    State.webcamStream = null;
  }

  const video = document.getElementById('webcamVideo');
  if (video) video.srcObject = null;

  document.getElementById('webcamContainer').style.display = 'none';
  document.getElementById('webcamBtn').innerHTML =
    '<i class="fas fa-video"></i><span data-en="Live Webcam" data-hi="लाइव वेबकैम">Live Webcam</span>';
}

function captureWebcam() {
  const video  = document.getElementById('webcamVideo');
  const canvas = document.getElementById('webcamCanvas');
  const ctx    = canvas.getContext('2d');

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  // Convert canvas to blob → File
  canvas.toBlob((blob) => {
    const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
    processFile(file);
    stopWebcam();
    showNotification('Image captured from webcam!', 'success');
  }, 'image/jpeg', 0.92);
}

// ─────────────────────────────────────────────
// ANALYZE (MAIN FUNCTION)
// ─────────────────────────────────────────────
async function analyzeImage() {
  if (!State.selectedFile) {
    showNotification(t('uploadFirst'), 'warning');
    return;
  }

  // Show loading state
  setAnalyzeButtonLoading(true);
  showLoadingState();

  try {
    // Build form data
    const formData = new FormData();
    formData.append('image', State.selectedFile);

    // Call API
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Server error: ${response.status}`);
    }

    // Display result
    displayResult(data);
    State.lastResult = data;

  } catch (err) {
    console.error('Analysis error:', err);
    displayError(err.message || t('networkError'));
  } finally {
    setAnalyzeButtonLoading(false);
  }
}

// ─────────────────────────────────────────────
// DISPLAY RESULTS
// ─────────────────────────────────────────────
function displayResult(data) {
  const {
    disease_name, confidence_score, description,
    treatment, prevention, severity, is_healthy,
    processing_time
  } = data;

  // Show results section, hide loading, show result card
  document.getElementById('resultsSection').style.display = 'flex';
  document.getElementById('loadingCard').style.display   = 'none';
  document.getElementById('resultCard').style.display    = 'block';
  document.getElementById('errorCard').style.display     = 'none';

  // ── Header ────────────────────────────────────────────
  const header = document.getElementById('resultHeader');
  header.className = `result-header ${is_healthy ? 'healthy' : 'diseased'}`;

  // Emoji
  document.getElementById('resultEmoji').textContent = is_healthy ? '✅' : '🔬';

  // Icon wrapper color
  const iconWrapper = document.getElementById('resultIconWrapper');
  iconWrapper.style.background = is_healthy
    ? 'rgba(34,197,94,0.12)'
    : 'rgba(245,158,11,0.12)';

  // Status badge
  const badge = document.getElementById('resultStatusBadge');
  badge.textContent = is_healthy ? '✓ Healthy Plant' : '⚠ Disease Detected';
  badge.className   = `result-status-badge ${is_healthy ? 'badge-healthy' : 'badge-diseased'}`;

  // Disease name
  document.getElementById('resultDiseaseName').textContent = disease_name;

  // ── Confidence Bar ────────────────────────────────────
  const confPercent = Math.round(confidence_score * 100);
  const confFill    = document.getElementById('confidenceBarFill');
  const confValue   = document.getElementById('confidenceValue');

  // Animate bar
  setTimeout(() => {
    confFill.style.width = `${confPercent}%`;

    // Color based on confidence
    if (confPercent >= 85) {
      confFill.style.background = 'linear-gradient(90deg, #16a34a, #22c55e)';
    } else if (confPercent >= 65) {
      confFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
    } else {
      confFill.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    }
  }, 100);

  // Animate number
  animateNumber(confValue, 0, confPercent, 1000, v => `${v}%`);

  // ── Severity ─────────────────────────────────────────
  const sev         = (severity || 'Mild').toLowerCase();
  const severityPill= document.getElementById('severityPill');
  const dots        = [document.getElementById('dot1'), document.getElementById('dot2'), document.getElementById('dot3')];

  severityPill.textContent = severity || 'Mild';
  severityPill.className   = `severity-pill severity-${sev}`;

  // Activate dots
  const dotCount = { mild: 1, moderate: 2, severe: 3 }[sev] || 1;
  dots.forEach((dot, i) => {
    dot.className = `severity-dot ${i < dotCount ? `active-${sev}` : ''}`;
  });

  // ── Processing Time ───────────────────────────────────
  document.getElementById('timeValue').textContent =
    processing_time ? `${processing_time}s` : '—';

  // ── Tab Content ───────────────────────────────────────
  const lang = State.currentLang; // 'en' or 'hi'
  document.getElementById('descriptionText').textContent = data[`description_${lang}`] || description || 'No description available.';
  document.getElementById('treatmentText').textContent   = data[`treatment_${lang}`] || treatment   || 'No treatment information available.';
  document.getElementById('preventionText').textContent  = data[`prevention_${lang}`] || prevention  || 'No prevention information available.';

  // Reset to first tab
  switchTab(document.querySelector('.tab-btn'), 'description');

  // Smooth scroll to result
  setTimeout(() => {
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

function displayError(message) {
  document.getElementById('resultsSection').style.display = 'flex';
  document.getElementById('loadingCard').style.display    = 'none';
  document.getElementById('resultCard').style.display     = 'none';
  document.getElementById('errorCard').style.display      = 'block';
  document.getElementById('errorMessage').textContent     = message;

  document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showLoadingState() {
  document.getElementById('resultsSection').style.display = 'flex';
  document.getElementById('loadingCard').style.display    = 'block';
  document.getElementById('resultCard').style.display     = 'none';
  document.getElementById('errorCard').style.display      = 'none';
}

function hideResults() {
  document.getElementById('resultsSection').style.display = 'none';
}

// ─────────────────────────────────────────────
// TAB SWITCHING
// ─────────────────────────────────────────────
function switchTab(btn, tabName) {
  // Deactivate all tabs
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

  // Activate selected
  btn.classList.add('active');
  const pane = document.getElementById(`tab-${tabName}`);
  if (pane) pane.classList.add('active');
}

// ─────────────────────────────────────────────
// SHARE RESULT
// ─────────────────────────────────────────────
async function shareResult() {
  if (!State.lastResult) return;

  const { disease_name, confidence_score } = State.lastResult;
  const shareText = t('shareText', disease_name, confidence_score);
  const shareData = {
    title: t('shareTitle'),
    text : shareText,
    url  : window.location.href,
  };

  // Try Web Share API first (mobile)
  if (navigator.share) {
    try {
      await navigator.share(shareData);
      return;
    } catch (e) {
      // User cancelled or API failed, fall through to clipboard
    }
  }

  // Fallback: copy to clipboard
  const textToCopy = `${shareText}\n${window.location.href}`;
  try {
    await navigator.clipboard.writeText(textToCopy);
    showNotification(t('copied'), 'success');
  } catch (e) {
    showNotification('Could not share. Please copy the URL manually.', 'warning');
  }
}

// ─────────────────────────────────────────────
// BUTTON STATE HELPERS
// ─────────────────────────────────────────────
function enableAnalyzeButton() {
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = false;
}

function disableAnalyzeButton() {
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
}

function setAnalyzeButtonLoading(isLoading) {
  const btn     = document.getElementById('analyzeBtn');
  const btnText = btn.querySelector('.btn-text');
  const loading = btn.querySelector('.btn-loading');
  const icon    = btn.querySelector('.fa-search-plus');

  if (isLoading) {
    btn.disabled        = true;
    if (btnText)  btnText.style.display  = 'none';
    if (loading)  loading.style.display  = 'flex';
    if (icon)     icon.style.display     = 'none';
  } else {
    btn.disabled        = false;
    if (btnText)  btnText.style.display  = '';
    if (loading)  loading.style.display  = 'none';
    if (icon)     icon.style.display     = '';
  }
}

// ─────────────────────────────────────────────
// NOTIFICATION TOAST
// ─────────────────────────────────────────────
function showNotification(message, type = 'info') {
  // Remove existing
  const existing = document.getElementById('notification-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id    = 'notification-toast';

  const icons = {
    success: '✓', error: '✕', warning: '⚠', info: 'ℹ'
  };
  const colors = {
    success: '#16a34a', error: '#ef4444', warning: '#f59e0b', info: '#3b82f6'
  };

  toast.style.cssText = `
    position: fixed;
    top: 80px;
    right: 20px;
    z-index: 9999;
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-left: 4px solid ${colors[type]};
    border-radius: 12px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    font-size: 0.9rem;
    font-weight: 500;
    max-width: 340px;
    animation: slideInRight 0.4s ease;
    backdrop-filter: blur(10px);
  `;

  toast.innerHTML = `
    <span style="font-size:1.1rem;color:${colors[type]}">${icons[type]}</span>
    <span>${message}</span>
    <button onclick="this.parentElement.remove()" style="
      margin-left:auto;background:none;border:none;
      color:var(--text-muted);cursor:pointer;font-size:1rem;padding:2px;
    ">×</button>
  `;

  document.body.appendChild(toast);

  // Auto remove after 4s
  setTimeout(() => {
    if (toast.parentNode) {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }
  }, 4000);
}

// ─────────────────────────────────────────────
// NUMBER ANIMATION
// ─────────────────────────────────────────────
function animateNumber(el, from, to, duration, formatter = v => v) {
  const start = performance.now();
  const update = (time) => {
    const elapsed  = time - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
    const current  = Math.round(from + (to - from) * eased);
    el.textContent = formatter(current);
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// ─────────────────────────────────────────────
// INTERSECTION OBSERVER (Scroll Animations)
// ─────────────────────────────────────────────
function initAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity   = '1';
          entry.target.style.transform = 'none';
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
  );

  // Animate step cards, plant tags
  document.querySelectorAll('.step-card, .plant-tag').forEach((el, i) => {
    el.style.opacity   = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition= `opacity 0.6s ease ${i * 0.08}s, transform 0.6s ease ${i * 0.08}s`;
    observer.observe(el);
  });
}

// ─────────────────────────────────────────────
// KEYBOARD SHORTCUTS
// ─────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  // Ctrl+Enter → Analyze
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    analyzeImage();
  }
  // Escape → Reset all
  if (e.key === 'Escape') {
    resetAll();
  }
});

// ─────────────────────────────────────────────
// CLICK ON DROP ZONE → Open File Dialog
// ─────────────────────────────────────────────
document.getElementById('dropZone').addEventListener('click', (e) => {
  // Don't trigger if clicking browse button or preview
  if (e.target.closest('.btn-browse') || e.target.closest('.preview-container')) return;
  if (document.getElementById('previewContainer').style.display === 'none' ||
      !document.getElementById('previewContainer').style.display) {
    document.getElementById('fileInput').click();
  }
});

// ─────────────────────────────────────────────
// INJECT TOAST ANIMATION CSS
// ─────────────────────────────────────────────
const style = document.createElement('style');
style.textContent = `
  @keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to   { opacity: 1; transform: none; }
  }
`;
document.head.appendChild(style);
