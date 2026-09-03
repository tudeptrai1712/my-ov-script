// OpenVINO GenAI Chat - Client Application

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const messagesContainer = document.getElementById('messagesContainer');
  const welcomeScreen = document.getElementById('welcomeScreen');
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const stopBtn = document.getElementById('stopBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const historyList = document.getElementById('historyList');
  const historySearch = document.getElementById('historySearch');
  const historyDirNotice = document.getElementById('historyDirNotice');

  // Universal Attachment Elements (Files & Images)
  const attachFileBtn = document.getElementById('attachFileBtn');
  const universalFileInput = document.getElementById('universalFileInput');
  const attachmentPreviewBar = document.getElementById('attachmentPreviewBar');
  const attachmentItemsList = document.getElementById('attachmentItemsList');

  // Top Nav Elements
  const navModelName = document.getElementById('navModelName');
  const navDevicePill = document.getElementById('navDevicePill');
  const navTypePill = document.getElementById('navTypePill');
  const navModelBtn = document.getElementById('navModelBtn');
  const navUnloadBtn = document.getElementById('navUnloadBtn');
  const navStatusDot = document.getElementById('navStatusDot');
  const reasoningPillBtn = document.getElementById('reasoningPillBtn');
  const reasoningPillState = document.getElementById('reasoningPillState');
  const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
  const sidebar = document.getElementById('sidebar');

  // Sidebar Status Elements
  const sidebarModelName = document.getElementById('sidebarModelName');
  const sidebarDeviceBadge = document.getElementById('sidebarDeviceBadge');
  const sidebarMemory = document.getElementById('sidebarMemory');
  const sidebarEjectBtn = document.getElementById('sidebarEjectBtn');
  const sidebarStatusDot = document.getElementById('sidebarStatusDot');

  // Settings Modal Elements
  const settingsModal = document.getElementById('settingsModal');
  const modalUnloadBtn = document.getElementById('modalUnloadBtn');
  const openSettingsBtn = document.getElementById('openSettingsBtn');
  const closeSettingsBtn = document.getElementById('closeSettingsBtn');
  const settingModelSelect = document.getElementById('settingModelSelect');
  const settingDeviceSelect = document.getElementById('settingDeviceSelect');
  const modelMetaTags = document.getElementById('modelMetaTags');
  const deviceNotice = document.getElementById('deviceNotice');
  const settingContext = document.getElementById('settingContext');
  const contextVal = document.getElementById('contextVal');
  const settingMaxTokens = document.getElementById('settingMaxTokens');
  const tokensVal = document.getElementById('tokensVal');
  const settingTemp = document.getElementById('settingTemp');
  const tempVal = document.getElementById('tempVal');
  const settingTopP = document.getElementById('settingTopP');
  const topPVal = document.getElementById('topPVal');
  const settingReasoning = document.getElementById('settingReasoning');
  const settingHistoryDir = document.getElementById('settingHistoryDir');
  const saveDefaultsBtn = document.getElementById('saveDefaultsBtn');
  const applyAndLoadBtn = document.getElementById('applyAndLoadBtn');
  const loadSpinner = document.getElementById('loadSpinner');
  const loadBtnText = document.getElementById('loadBtnText');

  // State
  let state = {
    models: [],
    loaded: false,
    modelName: '',
    device: 'GPU',
    isVlm: false,
    displayType: 'VLM',
    contextLength: 32768,
    maxNewTokens: 8192,
    reasoningEnabled: false,
    supportsReasoning: false,
    temperature: 0.7,
    topP: 0.95,
    stagedAttachments: [],
    messages: [],
    historyFiles: [],
    isGenerating: false,
    abortController: null,
    lastRequestPayload: null,
  };

  // ========================================================
  // INITIALIZATION
  // ========================================================

  async function init() {
    setupEventListeners();
    await loadSettingsData();
    await fetchModels();
    await checkModelStatus();
    await fetchHistory();

    // Auto-poll GPU memory
    setInterval(updateMemoryWidget, 3000);
  }

  function setupEventListeners() {
    // Chat Input Auto-Resize & Submit
    chatInput.addEventListener('input', () => {
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
      }
    });

    chatForm.addEventListener('submit', handleSendMessage);
    stopBtn.addEventListener('click', handleStopGeneration);
    newChatBtn.addEventListener('click', startNewChat);
    clearChatBtn.addEventListener('click', startNewChat);

    // Quick prompt cards
    document.querySelectorAll('.quick-card').forEach(card => {
      card.addEventListener('click', () => {
        const prompt = card.getAttribute('data-prompt');
        if (prompt) {
          chatInput.value = prompt;
          chatInput.style.height = 'auto';
          chatForm.dispatchEvent(new Event('submit'));
        }
      });
    });

    // Reasoning Pill Toggle
    reasoningPillBtn.addEventListener('click', () => {
      if (!state.supportsReasoning) return;
      state.reasoningEnabled = !state.reasoningEnabled;
      updateReasoningUI();
    });

    // Sidebar toggle (mobile)
    toggleSidebarBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    // History search
    historySearch.addEventListener('input', (e) => {
      renderHistoryList(e.target.value.toLowerCase());
    });

    // Modal Events
    openSettingsBtn.addEventListener('click', openSettings);
    navModelBtn.addEventListener('click', openSettings);
    closeSettingsBtn.addEventListener('click', closeSettings);
    settingsModal.addEventListener('click', (e) => {
      if (e.target === settingsModal) closeSettings();
    });

    settingModelSelect.addEventListener('change', () => {
      const selected = settingModelSelect.value;
      updateDeviceOptions(selected);
      updateModelMetaDisplay(selected);
    });

    // Sliders live text
    settingContext.addEventListener('input', (e) => {
      contextVal.textContent = Number(e.target.value).toLocaleString();
    });
    settingMaxTokens.addEventListener('input', (e) => {
      tokensVal.textContent = Number(e.target.value).toLocaleString();
    });
    settingTemp.addEventListener('input', (e) => {
      tempVal.textContent = e.target.value;
    });
    settingTopP.addEventListener('input', (e) => {
      topPVal.textContent = e.target.value;
    });

    saveDefaultsBtn.addEventListener('click', handleSaveDefaults);
    applyAndLoadBtn.addEventListener('click', handleApplyAndLoad);

    // Eject / Unload Model (LM Studio style)
    if (navUnloadBtn) {
      navUnloadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        handleUnloadModel();
      });
    }
    if (sidebarEjectBtn) {
      sidebarEjectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        handleUnloadModel();
      });
    }
    if (modalUnloadBtn) {
      modalUnloadBtn.addEventListener('click', handleUnloadModel);
    }

    // Universal File & Image Attachment
    if (attachFileBtn && universalFileInput) {
      attachFileBtn.addEventListener('click', () => {
        universalFileInput.click();
      });

      universalFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          handleFilesSelected(e.target.files);
          universalFileInput.value = '';
        }
      });
    }

    // Clipboard Paste (for files, images, or screenshots)
    chatInput.addEventListener('paste', (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      const filesToLoad = [];
      for (const item of items) {
        if (item.kind === 'file') {
          const file = item.getAsFile();
          if (file) filesToLoad.push(file);
        }
      }
      if (filesToLoad.length > 0) {
        handleFilesSelected(filesToLoad);
      }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(name => {
      chatForm.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        chatForm.classList.add('drag-over');
      }, false);
    });

    ['dragleave', 'drop'].forEach(name => {
      chatForm.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        chatForm.classList.remove('drag-over');
      }, false);
    });

    chatForm.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt?.files && dt.files.length > 0) {
        handleFilesSelected(dt.files);
      }
    });
  }

  function formatBytes(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function handleFilesSelected(fileList) {
    Array.from(fileList).forEach(file => {
      const isImg = file.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp|gif)$/i.test(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        const item = {
          id: 'att_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
          file: file,
          type: isImg ? 'image' : 'file',
          name: file.name,
          size: file.size,
          sizeHuman: formatBytes(file.size),
          data: e.target.result,
        };
        state.stagedAttachments.push(item);
        renderAttachmentPreviews();
      };
      reader.readAsDataURL(file);
    });
  }

  function renderAttachmentPreviews() {
    if (!attachmentPreviewBar || !attachmentItemsList) return;
    if (state.stagedAttachments.length === 0) {
      attachmentPreviewBar.style.display = 'none';
      attachmentItemsList.innerHTML = '';
      return;
    }

    attachmentPreviewBar.style.display = 'block';
    attachmentItemsList.innerHTML = '';

    state.stagedAttachments.forEach(item => {
      const el = document.createElement('div');
      el.className = 'attachment-item';

      if (item.type === 'image') {
        const img = document.createElement('img');
        img.className = 'attachment-thumb';
        img.src = item.data;
        img.alt = item.name;
        el.appendChild(img);
      } else {
        const icon = document.createElement('div');
        icon.className = 'attachment-file-icon';
        const ext = item.name.split('.').pop().toUpperCase().slice(0, 4);
        icon.textContent = ext || 'DOC';
        el.appendChild(icon);
      }

      const details = document.createElement('div');
      details.className = 'attachment-details';
      details.innerHTML = `
        <span class="attachment-name" title="${item.name}">${item.name}</span>
        <span class="attachment-size">${item.sizeHuman}</span>
      `;
      el.appendChild(details);

      const rmBtn = document.createElement('button');
      rmBtn.type = 'button';
      rmBtn.className = 'remove-attachment-btn';
      rmBtn.title = 'Remove attachment';
      rmBtn.innerHTML = '&times;';
      rmBtn.addEventListener('click', () => removeAttachment(item.id));
      el.appendChild(rmBtn);

      attachmentItemsList.appendChild(el);
    });
  }

  function removeAttachment(id) {
    state.stagedAttachments = state.stagedAttachments.filter(a => a.id !== id);
    renderAttachmentPreviews();
  }

  function clearStagedAttachments() {
    state.stagedAttachments = [];
    renderAttachmentPreviews();
    if (universalFileInput) universalFileInput.value = '';
  }

  // ========================================================
  // API CALLS
  // ========================================================

  async function loadSettingsData() {
    try {
      const res = await fetch('/api/settings');
      const data = await res.json();
      settingHistoryDir.value = data.history_dir || '';
      if (historyDirNotice) {
        historyDirNotice.textContent = data.history_dir ? 'Saved to: ' + data.history_dir : '';
      }
      state.reasoningEnabled = !!data.enable_reasoning;
      state.temperature = data.temperature ?? 0.7;
      state.topP = data.top_p ?? 0.95;
    } catch (err) {
      console.error('Failed to load settings:', err);
    }
  }

  async function fetchModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      state.models = data.models || [];

      settingModelSelect.innerHTML = '';
      if (state.models.length === 0) {
        settingModelSelect.innerHTML = '<option value="">No models found in root</option>';
        return;
      }

      state.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name} (${m.size_human})`;
        settingModelSelect.appendChild(opt);
      });

      if (state.models.length > 0) {
        const first = state.models[0].name;
        updateDeviceOptions(first);
        updateModelMetaDisplay(first);
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  }

  async function updateDeviceOptions(modelName) {
    try {
      const url = modelName ? `/api/devices?model_name=${encodeURIComponent(modelName)}` : '/api/devices';
      const res = await fetch(url);
      const data = await res.json();

      settingDeviceSelect.innerHTML = '';
      let warningText = '';

      (data.devices || []).forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.name;
        if (!d.compatible) {
          opt.disabled = true;
          opt.textContent = `${d.name} (Unavailable: ${d.reason})`;
          warningText += `${d.name}: ${d.reason}. `;
        } else {
          opt.textContent = d.name;
        }
        settingDeviceSelect.appendChild(opt);
      });

      // Default select first enabled option
      const firstEnabled = Array.from(settingDeviceSelect.options).find(o => !o.disabled);
      if (firstEnabled) {
        firstEnabled.selected = true;
      }

      deviceNotice.textContent = warningText ? `Note: ${warningText}` : '';
    } catch (err) {
      console.error('Failed to fetch device options:', err);
    }
  }

  function updateModelMetaDisplay(modelName) {
    const meta = state.models.find(m => m.name === modelName);
    modelMetaTags.innerHTML = '';
    if (!meta) return;

    const tags = [
      { text: meta.display_type, highlight: true },
      { text: `Precision: ${meta.precision}`, highlight: false },
      { text: meta.supports_reasoning ? 'Thinking: Supported' : 'Thinking: Not Supported', highlight: meta.supports_reasoning }
    ];

    tags.forEach(t => {
      const el = document.createElement('span');
      el.className = 'tag' + (t.highlight ? ' highlight' : '');
      el.textContent = t.text;
      modelMetaTags.appendChild(el);
    });

    // Bound context length slider
    if (meta.max_context) {
      settingContext.max = meta.max_context;
      if (Number(settingContext.value) > meta.max_context) {
        settingContext.value = meta.max_context;
        contextVal.textContent = meta.max_context.toLocaleString();
      }
    }
  }

  async function checkModelStatus() {
    try {
      const res = await fetch('/api/model/status');
      const data = await res.json();
      state.loaded = data.loaded;

      if (data.loaded) {
        state.modelName = data.model_name;
        state.device = data.device;
        state.isVlm = data.is_vlm;
        state.displayType = data.display_type;
        state.contextLength = data.context_length;
        state.maxNewTokens = data.max_new_tokens;
        state.supportsReasoning = data.supports_reasoning;
        state.reasoningEnabled = data.reasoning_enabled && data.supports_reasoning;

        navModelName.textContent = data.model_name;
        navDevicePill.textContent = data.device;
        navDevicePill.style.display = 'inline-block';
        navTypePill.textContent = data.display_type.includes('VLM') ? 'VLM' : 'LLM';
        navTypePill.style.display = 'inline-block';

        if (navStatusDot) navStatusDot.className = 'status-dot online';
        if (sidebarStatusDot) sidebarStatusDot.className = 'status-dot online';
        if (navUnloadBtn) navUnloadBtn.style.display = 'inline-flex';
        if (sidebarEjectBtn) sidebarEjectBtn.style.display = 'inline-block';
        if (modalUnloadBtn) modalUnloadBtn.style.display = 'inline-flex';

        sidebarModelName.textContent = data.model_name;
        sidebarDeviceBadge.textContent = data.device;
        sidebarMemory.textContent = data.gpu_memory_human;

        loadBtnText.textContent = 'Reload / Switch Model';
        chatInput.placeholder = 'Message OpenVINO... (Press Enter to send, paste/drop files or images)';

        updateReasoningUI();
      } else {
        navModelName.textContent = 'Select a model to load';
        navDevicePill.style.display = 'none';
        navTypePill.style.display = 'none';

        if (navStatusDot) navStatusDot.className = 'status-dot';
        if (sidebarStatusDot) sidebarStatusDot.className = 'status-dot';
        if (navUnloadBtn) navUnloadBtn.style.display = 'none';
        if (sidebarEjectBtn) sidebarEjectBtn.style.display = 'none';
        if (modalUnloadBtn) modalUnloadBtn.style.display = 'none';

        sidebarModelName.textContent = 'No Model Loaded';
        sidebarDeviceBadge.textContent = 'None';
        sidebarMemory.textContent = data.gpu_memory_human;

        loadBtnText.textContent = 'Apply & Load Model';
        chatInput.placeholder = 'No model loaded. Click here or select a model above to start chatting.';

        updateReasoningUI();
      }
    } catch (err) {
      console.error('Failed to get status:', err);
    }
  }

  async function handleUnloadModel() {
    try {
      if (modalUnloadBtn) modalUnloadBtn.disabled = true;
      if (navUnloadBtn) navUnloadBtn.disabled = true;
      if (sidebarEjectBtn) sidebarEjectBtn.disabled = true;

      const res = await fetch('/api/model/unload', { method: 'POST' });
      if (res.ok) {
        closeSettings();
        await checkModelStatus();
      }
    } catch (err) {
      console.error('Failed to unload model:', err);
    } finally {
      if (modalUnloadBtn) modalUnloadBtn.disabled = false;
      if (navUnloadBtn) navUnloadBtn.disabled = false;
      if (sidebarEjectBtn) sidebarEjectBtn.disabled = false;
    }
  }

  async function updateMemoryWidget() {
    if (!state.loaded) return;
    try {
      const res = await fetch('/api/model/status');
      const data = await res.json();
      sidebarMemory.textContent = data.gpu_memory_human;
    } catch (_) {}
  }

  function updateReasoningUI() {
    if (!state.supportsReasoning) {
      reasoningPillBtn.style.display = 'none';
      return;
    }
    reasoningPillBtn.style.display = 'flex';
    if (state.reasoningEnabled) {
      reasoningPillBtn.classList.add('active');
      reasoningPillState.textContent = 'ON';
    } else {
      reasoningPillBtn.classList.remove('active');
      reasoningPillState.textContent = 'OFF';
    }
  }

  // ========================================================
  // SETTINGS MODAL ACTIONS
  // ========================================================

  function openSettings() {
    settingsModal.classList.add('open');
    if (state.modelName) {
      settingModelSelect.value = state.modelName;
      updateDeviceOptions(state.modelName);
      updateModelMetaDisplay(state.modelName);
    }
    settingDeviceSelect.value = state.device || 'GPU';
    settingContext.value = state.contextLength;
    contextVal.textContent = Number(state.contextLength).toLocaleString();
    settingMaxTokens.value = state.maxNewTokens;
    tokensVal.textContent = Number(state.maxNewTokens).toLocaleString();
    settingTemp.value = state.temperature;
    tempVal.textContent = state.temperature;
    settingTopP.value = state.topP;
    topPVal.textContent = state.topP;
    settingReasoning.checked = state.reasoningEnabled;
  }

  function closeSettings() {
    settingsModal.classList.remove('open');
  }

  async function handleSaveDefaults() {
    const payload = {
      selected_model: settingModelSelect.value,
      selected_device: settingDeviceSelect.value,
      context_length: Number(settingContext.value),
      max_new_tokens: Number(settingMaxTokens.value),
      temperature: Number(settingTemp.value),
      top_p: Number(settingTopP.value),
      enable_reasoning: settingReasoning.checked,
    };

    try {
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      saveDefaultsBtn.textContent = 'Saved!';
      setTimeout(() => { saveDefaultsBtn.textContent = 'Save as Default'; }, 1500);
    } catch (err) {
      alert('Failed to save settings: ' + err.message);
    }
  }

  async function handleApplyAndLoad() {
    const modelName = settingModelSelect.value;
    const device = settingDeviceSelect.value;
    if (!modelName) {
      alert('Please select a model.');
      return;
    }

    loadSpinner.style.display = 'inline-block';
    loadBtnText.textContent = 'Compiling Model...';
    applyAndLoadBtn.disabled = true;

    try {
      const res = await fetch('/api/model/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          device: device,
          context_length: Number(settingContext.value),
          max_new_tokens: Number(settingMaxTokens.value),
          temperature: Number(settingTemp.value),
          top_p: Number(settingTopP.value),
          enable_reasoning: settingReasoning.checked,
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to load model');
      }

      await checkModelStatus();
      closeSettings();
    } catch (err) {
      alert('Error loading model: ' + err.message);
    } finally {
      loadSpinner.style.display = 'none';
      loadBtnText.textContent = 'Apply & Load Model';
      applyAndLoadBtn.disabled = false;
    }
  }

  // ========================================================
  // CHAT INTERACTION & SSE STREAMING
  // ========================================================

  function startNewChat() {
    clearStagedAttachments();
    state.messages = [];
    messagesContainer.innerHTML = '';
    messagesContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'block';
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.focus();
  }

  async function handleSendMessage(e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    if ((!text && state.stagedAttachments.length === 0) || state.isGenerating) return;

    if (!state.loaded) {
      openSettings();
      return;
    }

    // Capture staged attachments
    const currentAttachments = [...state.stagedAttachments];
    clearStagedAttachments();

    // Add user message
    const userDisplayPrompt = text || '(Attached files)';
    state.messages.push({ role: 'user', content: userDisplayPrompt });
    appendMessageElement('user', userDisplayPrompt, false, currentAttachments);
    chatInput.value = '';
    chatInput.style.height = 'auto';

    const images = currentAttachments.filter(a => a.type === 'image');
    const files = currentAttachments.filter(a => a.type === 'file');

    const payload = {
      messages: state.messages,
      enable_reasoning: state.reasoningEnabled,
      max_new_tokens: state.maxNewTokens,
      temperature: state.temperature,
      top_p: state.topP,
    };

    if (images.length > 0) {
      payload.images = images.map(a => a.data);
    }
    if (files.length > 0) {
      payload.files = files.map(a => ({
        filename: a.name,
        data: a.data,
        size: a.size,
      }));
    }

    state.lastRequestPayload = payload;
    await executeChatStream(payload);
  }

  async function executeChatStream(payload) {
    welcomeScreen.style.display = 'none';

    // Prepare assistant bubble
    const aiBubble = appendMessageElement('assistant', '', true);
    setGenerating(true);

    state.abortController = new AbortController();

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: state.abortController.signal,
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Generation failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let assistantFullText = '';
      let thinkingText = '';
      let inThinkingMode = false;
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // Keep partial line

        for (const block of lines) {
          if (!block.startsWith('data: ')) continue;
          const dataStr = block.replace(/^data: /, '').trim();
          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);

            if (data.chunk) {
              const chunk = data.chunk;

              // Check for thinking token markers
              if (chunk.includes('<|channel>thought') || chunk.includes('<think>')) {
                inThinkingMode = true;
              }

              if (inThinkingMode) {
                thinkingText += chunk.replace('<|channel>thought\n', '').replace('<think>', '');
                if (chunk.includes('<channel|>') || chunk.includes('</think>')) {
                  inThinkingMode = false;
                  thinkingText = thinkingText.replace('<channel|>', '').replace('</think>', '');
                }
                updateThinkingBlock(aiBubble, thinkingText);
              } else {
                assistantFullText += chunk;
                updateMessageContent(aiBubble, assistantFullText);
              }
            }

            if (data.done) {
              if (data.metrics) {
                renderMetrics(aiBubble, data.metrics);
              }
            }

            if (data.error) {
              updateMessageContent(aiBubble, `<span style="color: var(--danger-color);">Error: ${data.error}</span>`);
            }
          } catch (_) {}
        }
      }

      // Save turn locally
      state.messages.push({ role: 'assistant', content: assistantFullText });

      // Add Copy & Retry action buttons to output
      renderMessageActions(aiBubble, assistantFullText);

      await fetchHistory(); // Refresh sidebar history list
    } catch (err) {
      if (err.name !== 'AbortError') {
        updateMessageContent(aiBubble, `<span style="color: var(--danger-color);">Error: ${err.message}</span>`);
        renderMessageActions(aiBubble, '');
      }
    } finally {
      setGenerating(false);
      state.abortController = null;
    }
  }

  function renderMessageActions(row, fullText) {
    const content = row.querySelector('.message-content');
    if (!content) return;

    let actionsBar = content.querySelector('.message-actions');
    if (actionsBar) actionsBar.remove();

    actionsBar = document.createElement('div');
    actionsBar.className = 'message-actions';

    // Copy Button
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'msg-action-btn copy-msg-btn';
    copyBtn.title = 'Copy response to clipboard';
    copyBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span>Copy</span>
    `;
    copyBtn.addEventListener('click', () => {
      const rawText = fullText || row.querySelector('.text-body')?.innerText || '';
      const clean = rawText.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
      navigator.clipboard.writeText(clean || rawText).then(() => {
        const span = copyBtn.querySelector('span');
        copyBtn.classList.add('copied');
        if (span) span.textContent = 'Copied!';
        setTimeout(() => {
          copyBtn.classList.remove('copied');
          if (span) span.textContent = 'Copy';
        }, 2000);
      });
    });

    // Retry / Regenerate Button
    const retryBtn = document.createElement('button');
    retryBtn.type = 'button';
    retryBtn.className = 'msg-action-btn retry-msg-btn';
    retryBtn.title = 'Regenerate response';
    retryBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="23 4 23 10 17 10"></polyline>
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
      </svg>
      <span>Retry</span>
    `;
    retryBtn.addEventListener('click', () => {
      handleRetry(row);
    });

    actionsBar.appendChild(copyBtn);
    actionsBar.appendChild(retryBtn);
    content.appendChild(actionsBar);
  }

  async function handleRetry(row) {
    if (state.isGenerating) return;

    // Pop the assistant message if it was already recorded
    if (state.messages.length > 0 && state.messages[state.messages.length - 1].role === 'assistant') {
      state.messages.pop();
    }

    // Remove row from DOM
    if (row && row.parentNode) {
      row.remove();
    }

    // Reconstruct payload
    const payload = {
      ...(state.lastRequestPayload || {}),
      messages: state.messages,
      enable_reasoning: state.reasoningEnabled,
      max_new_tokens: state.maxNewTokens,
      temperature: state.temperature,
      top_p: state.topP,
    };

    await executeChatStream(payload);
  }

  function handleStopGeneration() {
    if (state.abortController) {
      state.abortController.abort();
      setGenerating(false);
    }
  }

  function setGenerating(isGen) {
    state.isGenerating = isGen;
    if (isGen) {
      sendBtn.style.display = 'none';
      stopBtn.style.display = 'flex';
    } else {
      sendBtn.style.display = 'flex';
      stopBtn.style.display = 'none';
    }
  }

  // ========================================================
  // DOM RENDERING HELPERS
  // ========================================================

  function appendMessageElement(role, text, isPending = false, attachments = null) {
    const row = document.createElement('div');
    row.className = `message-row ${role}-row`;

    const avatar = document.createElement('div');
    avatar.className = `avatar ${role}-avatar`;
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const content = document.createElement('div');
    content.className = 'message-content';

    if (attachments && attachments.length > 0) {
      const attBox = document.createElement('div');
      attBox.className = 'message-attachments';

      attachments.forEach(att => {
        if (att.type === 'image') {
          const img = document.createElement('img');
          img.className = 'user-message-image';
          img.src = att.data;
          img.alt = att.name;
          attBox.appendChild(img);
        } else {
          const card = document.createElement('div');
          card.className = 'message-file-card';
          card.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
            </svg>
            <span><b>${att.name}</b> (${att.sizeHuman})</span>
          `;
          attBox.appendChild(card);
        }
      });
      content.appendChild(attBox);
    }

    if (role === 'assistant' && isPending && state.reasoningEnabled) {
      const thinkBox = document.createElement('details');
      thinkBox.className = 'thinking-block';
      thinkBox.open = true;
      thinkBox.innerHTML = '<summary>Thinking Process...</summary><div class="thinking-text">...</div>';
      content.appendChild(thinkBox);
    }

    const textEl = document.createElement('div');
    textEl.className = 'text-body';
    textEl.innerHTML = formatMarkdown(text);
    content.appendChild(textEl);

    row.appendChild(avatar);
    row.appendChild(content);
    messagesContainer.appendChild(row);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return row;
  }

  function updateThinkingBlock(row, text) {
    let thinkBox = row.querySelector('.thinking-block');
    if (!thinkBox) {
      thinkBox = document.createElement('details');
      thinkBox.className = 'thinking-block';
      thinkBox.open = true;
      thinkBox.innerHTML = '<summary>Thinking Process</summary><div class="thinking-text"></div>';
      row.querySelector('.message-content').prepend(thinkBox);
    }
    const tBody = thinkBox.querySelector('.thinking-text');
    if (tBody) tBody.textContent = text;
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function updateMessageContent(row, text) {
    const textEl = row.querySelector('.text-body');
    if (textEl) {
      textEl.innerHTML = formatMarkdown(text);
    }
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function renderMetrics(row, metrics) {
    const content = row.querySelector('.message-content');
    if (!content) return;
    const bar = document.createElement('div');
    bar.className = 'message-metrics';

    const parts = [];
    if (metrics.output_tokens) parts.push(`${metrics.output_tokens} tokens`);
    if (metrics.throughput) parts.push(`${metrics.throughput} tok/s`);
    if (metrics.ttft) parts.push(`TTFT: ${metrics.ttft}ms`);

    bar.textContent = parts.join(' • ');
    content.appendChild(bar);
  }

  // Configure marked if loaded
  if (window.marked) {
    window.marked.setOptions({
      gfm: true,
      breaks: true,
      pedantic: false,
    });
  }

  window.copyCode = function(btn) {
    const pre = btn.closest('.code-container')?.querySelector('code');
    if (!pre) return;
    navigator.clipboard.writeText(pre.innerText).then(() => {
      const span = btn.querySelector('span');
      if (span) span.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        if (span) span.textContent = 'Copy';
        btn.classList.remove('copied');
      }, 2000);
    });
  };

  function formatMarkdown(text) {
    if (!text) return '';

    if (window.marked) {
      try {
        // Pre-process code blocks for custom container if needed
        let html = window.marked.parse(text);

        // Enhance pre blocks with code-container, language tag and copy button
        html = html.replace(/<pre><code class="language-([\w-]+)">([\s\S]*?)<\/code><\/pre>/gi, (match, lang, code) => {
          return `
            <div class="code-container">
              <div class="code-header">
                <span class="code-lang">${lang}</span>
                <button class="copy-btn" onclick="copyCode(this)" title="Copy code">
                  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <span>Copy</span>
                </button>
              </div>
              <pre><code class="language-${lang}">${code}</code></pre>
            </div>
          `;
        });

        // For generic pre without class
        html = html.replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/gi, (match, code) => {
          return `
            <div class="code-container">
              <div class="code-header">
                <span class="code-lang">code</span>
                <button class="copy-btn" onclick="copyCode(this)" title="Copy code">
                  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <span>Copy</span>
                </button>
              </div>
              <pre><code>${code}</code></pre>
            </div>
          `;
        });

        return html;
      } catch (e) {
        console.error('Marked parsing error:', e);
      }
    }

    // Fallback basic parser
    let escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
    escaped = escaped.replace(/\*([^*]+)\*/g, '<i>$1</i>');
    escaped = escaped.replace(/\n\n/g, '<br><br>');
    escaped = escaped.replace(/\n/g, '<br>');
    return escaped;
  }

  // ========================================================
  // SAVED CHAT HISTORY IN SIDEBAR
  // ========================================================

  async function fetchHistory() {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      state.historyFiles = data.history || [];
      renderHistoryList();
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  }

  function renderHistoryList(filterQuery = '') {
    historyList.innerHTML = '';
    const filtered = state.historyFiles.filter(item => {
      return item.filename.toLowerCase().includes(filterQuery)
        || item.preview.toLowerCase().includes(filterQuery)
        || item.model.toLowerCase().includes(filterQuery);
    });

    if (filtered.length === 0) {
      historyList.innerHTML = '<div class="empty-history">No chat history found</div>';
      return;
    }

    filtered.forEach(item => {
      const div = document.createElement('div');
      div.className = 'history-item';

      const content = document.createElement('div');
      content.className = 'history-content';
      content.innerHTML = `<div>${item.preview || item.filename}</div><span class="history-meta">${item.model}</span>`;

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'history-delete-btn';
      deleteBtn.title = 'Delete Chat';
      deleteBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      `;

      deleteBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm(`Delete chat '${item.preview}'?`)) {
          await fetch(`/api/history/${encodeURIComponent(item.filename)}`, { method: 'DELETE' });
          await fetchHistory();
        }
      });

      div.appendChild(content);
      div.appendChild(deleteBtn);

      div.addEventListener('click', () => loadHistoricalChat(item.filename));

      historyList.appendChild(div);
    });
  }

  async function loadHistoricalChat(filename) {
    try {
      const res = await fetch(`/api/history/${encodeURIComponent(filename)}`);
      const data = await res.json();
      renderTranscript(data.content);
    } catch (err) {
      alert('Failed to load chat: ' + err.message);
    }
  }

  function renderTranscript(mdContent) {
    startNewChat();
    welcomeScreen.style.display = 'none';

    // Parse turns from markdown
    const turnBlocks = mdContent.split(/### Turn \d+ \([^\)]+\)/);
    turnBlocks.shift(); // remove header before Turn 1

    turnBlocks.forEach(block => {
      const userMatch = block.match(/\*\*User:\*\*\s*\n+([\s\S]*?)(?=\*\*AI|\n---|$)/);
      const aiMatch = block.match(/\*\*AI(?: `\[Thinking Mode\]`)?:\*\*\s*\n+([\s\S]*?)(?=> \*Metrics:|\n---|$)/);

      if (userMatch) {
        const uText = userMatch[1].trim();
        state.messages.push({ role: 'user', content: uText });
        appendMessageElement('user', uText);
      }
      if (aiMatch) {
        const aiText = aiMatch[1].trim();
        state.messages.push({ role: 'assistant', content: aiText });
        const aiBubble = appendMessageElement('assistant', aiText);
        renderMessageActions(aiBubble, aiText);
      }
    });
  }

  init();
});
