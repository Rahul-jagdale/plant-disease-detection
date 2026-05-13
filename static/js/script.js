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

  // Agar result already show ho raha hai toh re-fetch karo naye language mein
  if (State.lastResult) {
    refreshResultLanguage(lang);
  }
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

// Language change hone par result dobara fetch karo
async function refreshResultLanguage(lang) {
  const lastResult = State.lastResult;
  if (!lastResult) return;

  // Loading dikhao tabs mein
  document.getElementById('descriptionText').textContent = lang === 'hi' ? 'अनुवाद हो रहा है...' : 'Translating...';
  document.getElementById('treatmentText').textContent   = lang === 'hi' ? 'अनुवाद हो रहा है...' : 'Translating...';
  document.getElementById('preventionText').textContent  = lang === 'hi' ? 'अनुवाद हो रहा है...' : 'Translating...';

  // Badge bhi update karo
  const badge = document.getElementById('resultStatusBadge');
  if (badge) {
    badge.textContent = lastResult.is_healthy
      ? (lang === 'hi' ? '✓ स्वस्थ पौधा' : '✓ Healthy Plant')
      : (lang === 'hi' ? '⚠ बीमारी मिली' : '⚠ Disease Detected');
  }

  try {
    const formData = new FormData();
    formData.append('image', State.selectedFile);
    formData.append('lang', lang);

    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (response.ok) {
      // Naye language mein text update karo
      State.lastResult = data;
      document.getElementById('descriptionText').textContent = data.description || '';
      document.getElementById('treatmentText').textContent   = data.treatment   || '';
      document.getElementById('preventionText').textContent  = data.prevention  || '';
      document.getElementById('resultDiseaseName').textContent = data.disease_name;
    }
  } catch (err) {
    console.error('Language refresh failed:', err);
    document.getElementById('descriptionText').textContent = lastResult.description || '';
    document.getElementById('treatmentText').textContent   = lastResult.treatment   || '';
    document.getElementById('preventionText').textContent  = lastResult.prevention  || '';
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
    formData.append('lang', State.currentLang);

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
    showVoiceButton();

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
  const isHindi = State.currentLang === 'hi';
  badge.textContent = is_healthy
    ? (isHindi ? '✓ स्वस्थ पौधा' : '✓ Healthy Plant')
    : (isHindi ? '⚠ बीमारी मिली' : '⚠ Disease Detected');
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

// ── VOICE OUTPUT ────────────────────────────────────────────
function speakResult() {
    const lastResult = State.lastResult;
    if (!lastResult) return;

    const isHindi = State.currentLang === 'hi';
    const btn = document.getElementById('voiceBtn');

    // Already bol raha hai toh band karo
    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        btn.innerHTML = '🔊 <span data-en="Listen" data-hi="सुनिए">Listen</span>';
        btn.onclick = speakResult;
        return;
    }

    const text = isHindi
        ? `पौधे की जांच हो गई है। बीमारी का नाम है: ${lastResult.disease_name}। ${lastResult.description}। इलाज: ${lastResult.treatment}। बचाव: ${lastResult.prevention}`
        : `Plant analysis complete. Disease: ${lastResult.disease_name}. ${lastResult.description}. Treatment: ${lastResult.treatment}. Prevention: ${lastResult.prevention}`;

    const utterance = new SpeechSynthesisUtterance(text);

    // Best voice dhundho
    const voices = window.speechSynthesis.getVoices();

    if (isHindi) {
        // Hindi voice dhundho
        const hindiVoice = voices.find(v => v.lang === 'hi-IN') ||
                           voices.find(v => v.lang.startsWith('hi'));
        if (hindiVoice) {
            utterance.voice = hindiVoice;
        }
        utterance.lang  = 'hi-IN';
        utterance.rate  = 0.8;
        utterance.pitch = 1.0;
    } else {
        // Best English voice dhundho — Google ya Microsoft prefer karo
        const goodVoice = voices.find(v => v.name.includes('Google') && v.lang === 'en-US') ||
                          voices.find(v => v.name.includes('Microsoft') && v.lang === 'en-US') ||
                          voices.find(v => v.lang === 'en-US');
        if (goodVoice) utterance.voice = goodVoice;
        utterance.lang  = 'en-US';
        utterance.rate  = 0.85;
        utterance.pitch = 1.0;
    }

    utterance.volume = 1.0;

    utterance.onstart = () => {
        btn.innerHTML = '⏹️ <span>Stop</span>';
        btn.onclick = speakResult;
    };

    utterance.onend = () => {
        btn.innerHTML = '🔊 <span data-en="Listen" data-hi="सुनिए">Listen</span>';
        btn.onclick = speakResult;
    };

    // Voices load hone ka wait karo
    if (voices.length === 0) {
        window.speechSynthesis.onvoiceschanged = () => {
            window.speechSynthesis.speak(utterance);
        };
    } else {
        window.speechSynthesis.speak(utterance);
    }
}
// Result aane ke baad voice button dikhao
function showVoiceButton() {
    const btn = document.getElementById('voiceBtn');
    if (btn) btn.style.display = 'inline-flex';
}

// ─────────────────────────────────────────────
// WEATHER DASHBOARD
// ─────────────────────────────────────────────

// Weather Icons Mapping (WMO codes to Emojis)
const weatherIcons = {
  0: '☀️', // Clear sky
  1: '🌤️', // Mainly clear
  2: '⛅', // Partly cloudy
  3: '☁️', // Overcast
  45: '🌫️', // Fog
  48: '🌫️', // Depositing rime fog
  51: '🌧️', // Drizzle: Light
  53: '🌧️', // Drizzle: Moderate
  55: '🌧️', // Drizzle: Dense
  61: '🌧️', // Rain: Slight
  63: '🌧️', // Rain: Moderate
  65: '🌧️', // Rain: Heavy
  71: '❄️', // Snow fall: Slight
  73: '❄️', // Snow fall: Moderate
  75: '❄️', // Snow fall: Heavy
  95: '⛈️', // Thunderstorm: Slight or moderate
  96: '⛈️', // Thunderstorm with slight hail
  99: '⛈️'  // Thunderstorm with heavy hail
};

const weatherDescriptionsEn = {
  0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Fog', 48: 'Depositing rime fog', 51: 'Light Drizzle', 53: 'Moderate Drizzle',
  55: 'Dense Drizzle', 61: 'Slight Rain', 63: 'Moderate Rain', 65: 'Heavy Rain',
  71: 'Slight Snow', 73: 'Moderate Snow', 75: 'Heavy Snow',
  95: 'Thunderstorm', 96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'
};

const weatherDescriptionsHi = {
  0: 'साफ आसमान', 1: 'मुख्यतः साफ', 2: 'आंशिक बादल', 3: 'बादल छाए रहेंगे',
  45: 'कोहरा', 48: 'घना कोहरा', 51: 'हल्की बूंदाबांदी', 53: 'मध्यम बूंदाबांदी',
  55: 'घनी बूंदाबांदी', 61: 'हल्की बारिश', 63: 'मध्यम बारिश', 65: 'भारी बारिश',
  71: 'हल्की बर्फबारी', 73: 'मध्यम बर्फबारी', 75: 'भारी बर्फबारी',
  95: 'आंधी तूफान', 96: 'आंधी तूफान और ओले', 99: 'आंधी तूफान और भारी ओले'
};

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Weather
  const searchWeatherBtn = document.getElementById('searchWeatherBtn');
  const locationWeatherBtn = document.getElementById('locationWeatherBtn');
  const weatherCityInput = document.getElementById('weatherCityInput');

  if(searchWeatherBtn) {
    searchWeatherBtn.addEventListener('click', () => {
      if(weatherCityInput.value.trim()) {
        getWeatherByCity(weatherCityInput.value.trim());
      }
    });
  }

  if(weatherCityInput) {
    weatherCityInput.addEventListener('keypress', (e) => {
      if(e.key === 'Enter' && weatherCityInput.value.trim()) {
        getWeatherByCity(weatherCityInput.value.trim());
      }
    });
  }

  if(locationWeatherBtn) {
    locationWeatherBtn.addEventListener('click', getWeatherByLocation);
  }

  // Load default weather on load
  getWeatherByCity("New Delhi");
});

function showWeatherLoading() {
  document.getElementById('weatherDashboard').style.display = 'none';
  document.getElementById('weatherError').style.display = 'none';
  document.getElementById('weatherLoading').style.display = 'block';
}

function hideWeatherLoading() {
  document.getElementById('weatherLoading').style.display = 'none';
}

function showWeatherError() {
  hideWeatherLoading();
  document.getElementById('weatherDashboard').style.display = 'none';
  document.getElementById('weatherError').style.display = 'block';
}

async function getWeatherByLocation() {
  if (!navigator.geolocation) {
    showNotification("Geolocation is not supported by your browser", "error");
    return;
  }
  showWeatherLoading();
  navigator.geolocation.getCurrentPosition(
    async (position) => {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;
      await fetchWeatherData(lat, lon, State.currentLang === 'hi' ? "मेरा स्थान" : "My Location");
    },
    (error) => {
      showWeatherError();
      showNotification("Could not get your location.", "error");
    }
  );
}

async function getWeatherByCity(city) {
  showWeatherLoading();
  try {
    const geoRes = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`);
    const geoData = await geoRes.json();

    if (!geoData.results || geoData.results.length === 0) {
      throw new Error("City not found");
    }
    const { latitude, longitude, name, admin1 } = geoData.results[0];
    const locationName = admin1 ? `${name}, ${admin1}` : name;
    await fetchWeatherData(latitude, longitude, locationName);
  } catch (err) {
    console.error("City Search Error:", err);
    showWeatherError();
    showNotification("City not found. Please try another name.", "warning");
  }
}

async function fetchWeatherData(lat, lon, locationName) {
  try {
    // Fetch Weather
    const weatherRes = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m,visibility&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto`);
    const weatherData = await weatherRes.json();

    // Fetch AQI
    const aqiRes = await fetch(`https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${lat}&longitude=${lon}&current=european_aqi`);
    const aqiData = await aqiRes.json();

    updateWeatherUI(weatherData, aqiData, locationName);
  } catch (err) {
    console.error("Weather Fetch Error:", err);
    showWeatherError();
  }
}

function updateWeatherUI(weather, aqi, locationName) {
  hideWeatherLoading();

  const current = weather.current;
  const daily = weather.daily;
  const isHindi = State.currentLang === 'hi';

  // Header
  document.getElementById('weatherCityName').textContent = locationName;
  const dateOpts = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
  document.getElementById('weatherDate').textContent = new Date().toLocaleDateString(isHindi ? 'hi-IN' : 'en-US', dateOpts);

  // Main Card
  document.getElementById('currentTemp').textContent = Math.round(current.temperature_2m);
  document.getElementById('currentWeatherIcon').textContent = weatherIcons[current.weather_code] || '🌤️';

  const descDict = isHindi ? weatherDescriptionsHi : weatherDescriptionsEn;
  document.getElementById('currentWeatherDesc').textContent = descDict[current.weather_code] || (isHindi ? 'अज्ञात' : 'Unknown');

  // Metrics
  document.getElementById('windSpeed').textContent = `${current.wind_speed_10m} km/h`;
  document.getElementById('humidity').textContent = `${current.relative_humidity_2m}%`;
  document.getElementById('visibility').textContent = `${(current.visibility / 1000).toFixed(1)} km`;

  const aqiVal = aqi.current ? aqi.current.european_aqi : '--';
  document.getElementById('aqi').textContent = aqiVal;

  const aqiEl = document.getElementById('aqi');
  if(aqiVal !== '--') {
    if(aqiVal <= 20) aqiEl.style.color = '#22c55e'; // Good
    else if(aqiVal <= 50) aqiEl.style.color = '#eab308'; // Fair
    else if(aqiVal <= 100) aqiEl.style.color = '#f97316'; // Poor
    else aqiEl.style.color = '#ef4444'; // Very poor
  }

  // 7-Day Forecast
  const grid = document.getElementById('forecastGrid');
  grid.innerHTML = '';
  for(let i = 0; i < 7; i++) {
    const date = new Date(daily.time[i]);
    const dayName = date.toLocaleDateString(isHindi ? 'hi-IN' : 'en-US', { weekday: 'short' });
    const wCode = daily.weather_code[i];
    const wIcon = weatherIcons[wCode] || '🌤️';
    const tMax = Math.round(daily.temperature_2m_max[i]);
    const tMin = Math.round(daily.temperature_2m_min[i]);

    const card = document.createElement('div');
    card.className = 'forecast-card';
    card.innerHTML = `
      <span class="fc-day">${i === 0 ? (isHindi ? 'आज' : 'Today') : dayName}</span>
      <span class="fc-icon">${wIcon}</span>
      <div class="fc-temp">
        <span class="fc-high">${tMax}°</span>
        <span class="fc-low">${tMin}°</span>
      </div>
    `;
    grid.appendChild(card);
  }

  // Show
  document.getElementById('weatherDashboard').style.display = 'grid';
}

// Hook into existing language change
const originalApplyLanguage = applyLanguage;
applyLanguage = function(lang) {
  originalApplyLanguage(lang);
  // Re-translate placeholders manually if needed
  const input = document.getElementById('weatherCityInput');
  if(input) {
    input.placeholder = input.getAttribute(`data-${lang}-placeholder`) || 'Enter city name...';
  }
  // Refresh weather text if already loaded
  const city = document.getElementById('weatherCityName')?.textContent;
  if(city && city !== 'City Name') {
    if(city === 'My Location' || city === 'मेरा स्थान') {
      getWeatherByLocation();
    } else {
      getWeatherByCity(city);
    }
  }
};

// ─────────────────────────────────────────────
// PDF REPORT GENERATION (using jsPDF directly)
// ─────────────────────────────────────────────
function downloadReport() {
  if (!State.lastResult) return;

  const data = State.lastResult;
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF('p', 'mm', 'a4');
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  const contentWidth = pageWidth - margin * 2;
  let y = 20;

  // Helper: wrap long text and handle page breaks
  function addWrappedText(text, x, startY, maxWidth, lineHeight) {
    const lines = doc.splitTextToSize(text || 'N/A', maxWidth);
    lines.forEach(function(line) {
      if (startY > 270) {
        doc.addPage();
        startY = 20;
      }
      doc.text(line, x, startY);
      startY += lineHeight;
    });
    return startY;
  }

  // ── Header (green banner) ───────────────────
  doc.setFillColor(22, 163, 74);
  doc.rect(0, 0, pageWidth, 40, 'F');

  doc.setTextColor(255, 255, 255);
  doc.setFontSize(22);
  doc.setFont('helvetica', 'bold');
  doc.text('PlantDoc AI Report', pageWidth / 2, 18, { align: 'center' });

  doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  var dateStr = new Date().toLocaleDateString('en-US', {
    year: 'numeric', month: 'long', day: 'numeric',
    hour: '2-digit', minute: '2-digit'
  });
  doc.text('Date: ' + dateStr, pageWidth / 2, 28, { align: 'center' });

  doc.setFontSize(9);
  doc.text('Powered by AI for Farmers Worldwide', pageWidth / 2, 35, { align: 'center' });

  y = 52;

  // ── Disease Name ────────────────────────────
  doc.setTextColor(220, 38, 38);
  doc.setFontSize(20);
  doc.setFont('helvetica', 'bold');
  doc.text(data.disease_name || 'Unknown Disease', margin, y);
  y += 10;

  // ── Status / Confidence / Severity Row ──────
  doc.setFillColor(243, 244, 246);
  doc.roundedRect(margin, y, contentWidth, 18, 3, 3, 'F');

  doc.setTextColor(55, 65, 81);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');

  var statusLabel = data.is_healthy ? 'Healthy' : 'Infected';
  var confPercent = Math.round(data.confidence_score * 100);
  var severity = data.severity || 'Mild';

  doc.text('Status: ', margin + 5, y + 11);
  doc.setFont('helvetica', 'normal');
  if (data.is_healthy) { doc.setTextColor(22, 163, 74); }
  else { doc.setTextColor(239, 68, 68); }
  doc.text(statusLabel, margin + 23, y + 11);

  doc.setTextColor(55, 65, 81);
  doc.setFont('helvetica', 'bold');
  doc.text('Confidence: ', margin + 60, y + 11);
  doc.setFont('helvetica', 'normal');
  doc.text(confPercent + '%', margin + 85, y + 11);

  doc.setFont('helvetica', 'bold');
  doc.text('Severity: ', margin + 110, y + 11);
  doc.setFont('helvetica', 'normal');
  doc.text(severity, margin + 130, y + 11);

  y += 28;

  // ── Description Section ─────────────────────
  doc.setDrawColor(22, 163, 74);
  doc.setLineWidth(0.8);
  doc.line(margin, y, margin + contentWidth, y);
  y += 8;

  doc.setTextColor(22, 163, 74);
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('Description', margin, y);
  y += 7;

  doc.setTextColor(55, 65, 81);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  var descText = data.description_en || data.description || 'No description available.';
  y = addWrappedText(descText, margin, y, contentWidth, 5);
  y += 6;

  // ── Treatment Section ───────────────────────
  doc.setDrawColor(22, 163, 74);
  doc.line(margin, y, margin + contentWidth, y);
  y += 8;

  doc.setTextColor(22, 163, 74);
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('Recommended Treatment', margin, y);
  y += 7;

  doc.setTextColor(55, 65, 81);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  var treatText = data.treatment_en || data.treatment || 'No treatment information available.';
  y = addWrappedText(treatText, margin, y, contentWidth, 5);
  y += 6;

  // ── Prevention Section ──────────────────────
  doc.setDrawColor(22, 163, 74);
  doc.line(margin, y, margin + contentWidth, y);
  y += 8;

  doc.setTextColor(22, 163, 74);
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('Prevention Tips', margin, y);
  y += 7;

  doc.setTextColor(55, 65, 81);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  var prevText = data.prevention_en || data.prevention || 'No prevention information available.';
  y = addWrappedText(prevText, margin, y, contentWidth, 5);
  y += 10;

  // ── Footer ──────────────────────────────────
  doc.setDrawColor(200, 200, 200);
  doc.line(margin, 282, margin + contentWidth, 282);
  doc.setTextColor(156, 163, 175);
  doc.setFontSize(8);
  doc.text('Generated by PlantDoc AI - Empowering Farmers Worldwide', pageWidth / 2, 288, { align: 'center' });

  // ── Save ────────────────────────────────────
  var cleanName = data.disease_name.replace(/[^a-zA-Z0-9_]/g, '_');
  doc.save('PlantDoc_Report_' + cleanName + '.pdf');

  showNotification('Report downloaded successfully!', 'success');
}

