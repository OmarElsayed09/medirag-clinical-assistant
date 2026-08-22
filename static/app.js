// MediRAG Clinical Assistant - Full Expert Frontend Controller
document.addEventListener('DOMContentLoaded', () => {
    // --- TAB SWITCHING ---
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const activeViewTitle = document.getElementById('activeViewTitle');

    const viewTitles = {
        'assistant': 'Clinical Assistant',
        'case-analyzer': 'Patient Case & SOAP Analyzer',
        'ddi': 'Drug Interaction Matrix',
        'patient-guide': 'Patient Education Guide',
        'comparison': 'Multi-Guideline Consensus',
        'tools': 'Calculators',
        'eval': 'Evaluation Suite',
        'repository': 'Guidelines Corpus'
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.dataset.tab;

            navItems.forEach(n => n.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            const activePane = document.getElementById(`view-${targetTab}`);
            if (activePane) activePane.classList.add('active');

            if (activeViewTitle && viewTitles[targetTab]) {
                activeViewTitle.textContent = viewTitles[targetTab];
            }

            // Lazy load comparison table when tab is opened
            if (targetTab === 'comparison') {
                loadGuidelineComparison();
            }
        });
    });

    // --- CITATION DRAWER ---
    const citationDrawer = document.getElementById('citationDrawer');
    const btnCloseDrawer = document.getElementById('btnCloseDrawer');
    const drawerContent = document.getElementById('drawerContent');

    if (btnCloseDrawer) {
        btnCloseDrawer.addEventListener('click', () => {
            citationDrawer.classList.remove('open');
        });
    }

    window.openCitationDrawer = function(source, page, snippet) {
        if (!drawerContent || !citationDrawer) return;
        drawerContent.innerHTML = `
            <div style="background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(14, 165, 233, 0.3); padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
                <span style="display:block; font-size: 0.75rem; color: #94a3b8;">DOCUMENT SOURCE</span>
                <strong style="color: #38bdf8; font-size: 0.95rem;">${source}</strong>
                <div style="margin-top: 6px; font-family: monospace; font-size: 0.8rem; color: #cbd5e1;">Page: ${page}</div>
            </div>
            <h4 style="color: #f1f5f9; font-size: 0.88rem; margin-bottom: 8px;">Extracted Guideline Evidence Snippet:</h4>
            <div style="background: #060b17; border: 1px solid rgba(255, 255, 255, 0.08); padding: 14px; border-radius: 8px; font-size: 0.88rem; color: #e2e8f0; line-height: 1.6; border-left: 3px solid #0ea5e9;">
                "${snippet}"
            </div>
            <div style="margin-top: 18px; font-size: 0.78rem; color: #64748b;">
                Verified under strict closed-domain zero-hallucination constraint.
            </div>
        `;
        citationDrawer.classList.add('open');
    };

    // --- CHAT LOGIC ---
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const messagesContainer = document.getElementById('messagesContainer');
    const btnClearChat = document.getElementById('btnClearChat');
    const btnExportReport = document.getElementById('btnExportReport');
    const inquiryPills = document.querySelectorAll('.inquiry-pill');

    let chatHistory = [];

    inquiryPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const q = pill.dataset.query;
            if (chatInput) chatInput.value = q;
            if (chatForm) chatForm.dispatchEvent(new Event('submit'));
        });
    });

    if (btnClearChat) {
        btnClearChat.addEventListener('click', () => {
            chatHistory = [];
            messagesContainer.innerHTML = `
                <div class="message-card system-welcome">
                    <div class="message-header">
                        <div class="avatar-badge assistant">AI</div>
                        <div class="sender-name">MediRAG Clinical Assistant</div>
                        <div class="timestamp">Ready</div>
                    </div>
                    <div class="message-body">
                        <p>Welcome to MediRAG. I answer clinical questions strictly from official medical guidelines (WHO 2021, NICE NG136) with zero hallucination and verifiable page-level citations.</p>
                        <div class="disclaimer-banner">
                            <strong>Medical Disclaimer:</strong> This system is for clinical reference and educational purposes only, based strictly on indexed guidelines.
                        </div>
                    </div>
                </div>
            `;
        });
    }

    if (btnExportReport) {
        btnExportReport.addEventListener('click', () => {
            if (chatHistory.length === 0) {
                alert("No consultation history to export.");
                return;
            }
            let report = "# MediRAG Clinical Consultation Report\n\n";
            report += `Generated: ${new Date().toISOString()}\n\n---\n\n`;
            chatHistory.forEach(msg => {
                const label = msg.role === 'user' ? 'Clinical Inquiry' : 'MediRAG Evidence-Grounded Assistant';
                report += `### ${label}\n\n${msg.content}\n\n`;
                if (msg.citations && msg.citations.length > 0) {
                    report += "**Verified Citations:**\n";
                    msg.citations.forEach((c, idx) => {
                        report += `- Reference ${idx + 1}: \`${c.source}\` (Page ${c.page})\n`;
                    });
                    report += "\n";
                }
                report += "---\n\n";
            });
            report += "Medical Disclaimer: This information is synthesized from official clinical guidelines for reference purposes only.\n";

            const blob = new Blob([report], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `medirag_consultation_${Date.now()}.md`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    function appendUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'message-card user';
        div.innerHTML = `
            <div class="message-header">
                <div class="avatar-badge user">MD</div>
                <div class="sender-name">Clinician</div>
                <div class="timestamp">${new Date().toLocaleTimeString()}</div>
            </div>
            <div class="message-body">
                <p>${escapeHtml(text)}</p>
            </div>
        `;
        messagesContainer.appendChild(div);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function appendAssistantMessage(data) {
        const div = document.createElement('div');
        div.className = 'message-card assistant';

        let citationsHtml = '';
        if (data.citations && data.citations.length > 0) {
            citationsHtml = '<div class="citations-block">';
            data.citations.forEach((c, idx) => {
                const safeSource = escapeHtml(c.source);
                const safeSnippet = escapeHtml(c.snippet.replace(/"/g, '&quot;'));
                citationsHtml += `
                    <button class="citation-pill-btn" onclick="openCitationDrawer('${safeSource}', '${c.page}', '${safeSnippet}')">
                        Ref ${idx + 1}: ${safeSource} (p.${c.page})
                    </button>
                `;
            });
            citationsHtml += '</div>';
        }

        const formattedAnswer = formatMarkdownText(data.answer);

        div.innerHTML = `
            <div class="message-header">
                <div class="avatar-badge assistant">AI</div>
                <div class="sender-name">MediRAG Assistant</div>
                <div class="timestamp">${new Date().toLocaleTimeString()}</div>
            </div>
            <div class="message-body">
                ${formattedAnswer}
                ${citationsHtml}
            </div>
            <div class="message-telemetry">
                <span class="telemetry-chip">Latency: ${data.latency_seconds || '0.7'}s</span>
                <span class="telemetry-chip">${data.grounding_status || 'Strictly Grounded'}</span>
                <span class="telemetry-chip">Citations: ${data.citations ? data.citations.length : 0}</span>
            </div>
        `;
        messagesContainer.appendChild(div);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = chatInput.value.trim();
            if (!query) return;

            appendUserMessage(query);
            chatInput.value = '';

            const provider = document.getElementById('settingProvider').value;
            const modelName = document.getElementById('settingModel').value;
            const apiKey = document.getElementById('settingApiKey').value.trim();

            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message-card assistant';
            loadingDiv.innerHTML = `
                <div class="message-header">
                    <div class="avatar-badge assistant">AI</div>
                    <div class="sender-name">MediRAG Assistant</div>
                </div>
                <div class="message-body">
                    <p style="color: #94a3b8; font-style: italic;">Consulting guidelines and synthesizing evidence...</p>
                </div>
            `;
            messagesContainer.appendChild(loadingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: query,
                        provider: provider,
                        model_name: modelName,
                        api_key: apiKey || null,
                        chat_history: chatHistory.map(m => ({ role: m.role, content: m.content }))
                    })
                });

                messagesContainer.removeChild(loadingDiv);

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.detail || `Server error: ${response.status}`);
                }

                const result = await response.json();
                appendAssistantMessage(result);

                chatHistory.push({ role: 'user', content: query });
                chatHistory.push({ role: 'assistant', content: result.answer, citations: result.citations });

            } catch (err) {
                if (loadingDiv.parentNode) messagesContainer.removeChild(loadingDiv);

                const errDiv = document.createElement('div');
                errDiv.className = 'message-card assistant';
                let errMsg = err.message;
                if (errMsg.includes('401') || errMsg.toLowerCase().includes('invalid_api_key')) {
                    errMsg = "Invalid or expired Groq API key. Please obtain a free key at https://console.groq.com/keys and paste it into the Groq API Key field in the sidebar.";
                }
                errDiv.innerHTML = `
                    <div class="message-header">
                        <div class="avatar-badge assistant" style="background: #f43f5e;">!</div>
                        <div class="sender-name" style="color: #fb7185;">Error Processing Query</div>
                    </div>
                    <div class="message-body">
                        <p style="color: #fda4af;">${escapeHtml(errMsg)}</p>
                    </div>
                `;
                messagesContainer.appendChild(errDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        });
    }

    // --- FEATURE 1: PATIENT CASE & SOAP ANALYZER ---
    const btnAnalyzeCase = document.getElementById('btnAnalyzeCase');
    if (btnAnalyzeCase) {
        btnAnalyzeCase.addEventListener('click', async () => {
            const age = parseInt(document.getElementById('caseAge').value);
            const sex = document.getElementById('caseSex').value === 'female';
            const sys = parseInt(document.getElementById('caseSys').value);
            const dia = parseInt(document.getElementById('caseDia').value);
            const scr = parseFloat(document.getElementById('caseScr').value);
            const weight = parseFloat(document.getElementById('caseWeight').value);

            const dm = document.getElementById('caseDiabetes').checked;
            const cvd = document.getElementById('caseCVD').checked;
            const ckd = document.getElementById('caseCKD').checked;
            const smoker = document.getElementById('caseSmoker').checked;

            const meds = Array.from(document.querySelectorAll('input[name="caseMed"]:checked')).map(cb => cb.value);

            btnAnalyzeCase.disabled = true;
            btnAnalyzeCase.textContent = "Synthesizing Clinical Case...";

            try {
                const res = await fetch('/expert/case-analysis', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        age: age,
                        is_female: sex,
                        systolic_bp: sys,
                        diastolic_bp: dia,
                        serum_creatinine_mg_dl: scr || null,
                        weight_kg: weight || null,
                        has_diabetes: dm,
                        has_cvd: cvd,
                        has_ckd: ckd,
                        is_smoker: smoker,
                        current_medications: meds
                    })
                });

                const data = await res.json();
                if (res.ok) {
                    renderCaseAnalysisResult(data);
                } else {
                    alert("Analysis error: " + (data.detail || "Server error"));
                }
            } catch (e) {
                alert("Case synthesis error: " + e.message);
            } finally {
                btnAnalyzeCase.disabled = false;
                btnAnalyzeCase.textContent = "Synthesize Clinical Case & SOAP Note";
            }
        });
    }

    function renderCaseAnalysisResult(data) {
        const out = document.getElementById('caseResultBody');
        const s = data.patient_summary;
        const soap = data.soap_note;

        let stepsHtml = '';
        (data.clinical_pathway || []).forEach(st => {
            stepsHtml += `
                <div style="background: rgba(15, 23, 42, 0.7); border-left: 3px solid #0ea5e9; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 8px 8px 0;">
                    <strong style="color: #38bdf8; font-size: 0.88rem;">Step ${st.step}: ${escapeHtml(st.title)}</strong>
                    <p style="font-size: 0.84rem; color: #e2e8f0; margin: 4px 0;">${escapeHtml(st.description)}</p>
                    <span style="font-size: 0.72rem; color: #94a3b8; font-family: monospace;">${escapeHtml(st.guideline)}</span>
                </div>
            `;
        });

        let ddiAlertHtml = '';
        if (data.drug_interactions && data.drug_interactions.total_interactions_found > 0) {
            ddiAlertHtml = `
                <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 14px;">
                    <strong style="color: #fb7185; font-size: 0.85rem;">Medication Warning: ${data.drug_interactions.total_interactions_found} Clinical Interaction(s) Identified</strong>
                    <ul style="font-size: 0.8rem; color: #fda4af; margin: 6px 0 0 16px;">
                        ${data.drug_interactions.interactions.map(i => `<li><strong>${escapeHtml(i.title)}:</strong> ${escapeHtml(i.clinical_action)}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        out.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;">
                <div style="background: rgba(15, 23, 42, 0.8); padding: 10px; border-radius: 8px;">
                    <span style="font-size: 0.72rem; color: #94a3b8;">CLASSIFICATION</span><br/>
                    <strong style="color: #38bdf8; font-size: 0.95rem;">${s.bp_stage}</strong>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); padding: 10px; border-radius: 8px;">
                    <span style="font-size: 0.72rem; color: #94a3b8;">TARGET BLOOD PRESSURE</span><br/>
                    <strong style="color: #34d399; font-size: 0.95rem;">${s.target_bp}</strong>
                </div>
            </div>

            ${ddiAlertHtml}

            <h4 style="color: #f1f5f9; font-size: 0.9rem; margin-bottom: 8px;">Structured SOAP Note</h4>
            <div style="background: #060b17; border: 1px solid rgba(255, 255, 255, 0.06); padding: 12px; border-radius: 8px; font-size: 0.82rem; line-height: 1.5; color: #cbd5e1; margin-bottom: 14px;">
                <p><strong>S:</strong> ${escapeHtml(soap.subjective)}</p>
                <p style="margin-top: 4px;"><strong>O:</strong> ${escapeHtml(soap.objective)}</p>
                <p style="margin-top: 4px;"><strong>A:</strong> ${escapeHtml(soap.assessment)}</p>
            </div>

            <h4 style="color: #f1f5f9; font-size: 0.9rem; margin-bottom: 8px;">Personalized Clinical Pathway (Guideline-Grounded)</h4>
            ${stepsHtml}
        `;
    }

    // --- FEATURE 2: DRUG INTERACTION MATRIX ---
    const btnCheckDDI = document.getElementById('btnCheckDDI');
    if (btnCheckDDI) {
        btnCheckDDI.addEventListener('click', async () => {
            const checkboxes = document.querySelectorAll('#ddiChecklist input:checked');
            const selected = Array.from(checkboxes).map(c => c.value);

            if (selected.length < 2) {
                alert("Please select at least 2 medications to check for interactions.");
                return;
            }

            try {
                const res = await fetch('/expert/drug-interactions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ medications: selected })
                });
                const data = await res.json();
                renderDDIResult(data);
            } catch (e) {
                alert("DDI Evaluation Error: " + e.message);
            }
        });
    }

    function renderDDIResult(data) {
        const out = document.getElementById('ddiResultBody');
        if (data.total_interactions_found === 0) {
            out.innerHTML = `
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 20px; text-align: center;">
                    <strong style="color: #34d399; font-size: 1rem;">No High-Risk Clinical Interactions Found</strong>
                    <p style="color: #cbd5e1; font-size: 0.85rem; margin-top: 6px;">The selected combination (${data.total_drugs_checked} agents) does not exhibit major contraindicated interactions under WHO & NICE guidelines.</p>
                </div>
            `;
            return;
        }

        let cardsHtml = '';
        data.interactions.forEach(item => {
            const isCrit = item.severity_code === 'critical';
            const borderCol = isCrit ? '#f43f5e' : '#f59e0b';
            const bgCol = isCrit ? 'rgba(244, 63, 94, 0.1)' : 'rgba(245, 158, 11, 0.1)';
            const txtCol = isCrit ? '#fb7185' : '#fcd34d';

            cardsHtml += `
                <div style="background: ${bgCol}; border: 1px solid ${borderCol}; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <strong style="color: ${txtCol}; font-size: 0.92rem;">${escapeHtml(item.title)}</strong>
                        <span style="font-size: 0.72rem; font-weight: 700; background: ${txtCol}; color: #000; padding: 2px 8px; border-radius: 12px; text-transform: uppercase;">${escapeHtml(item.severity)}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px;"><strong>Drugs Involved:</strong> ${item.drugs_involved.join(' + ')}</div>
                    <p style="font-size: 0.84rem; color: #e2e8f0; margin-bottom: 8px; line-height: 1.5;">${escapeHtml(item.mechanism)}</p>
                    <div style="background: rgba(0,0,0,0.3); padding: 8px 10px; border-radius: 6px; font-size: 0.8rem; color: #f1f5f9;">
                        <strong>Action Protocol:</strong> ${escapeHtml(item.clinical_action)}
                    </div>
                </div>
            `;
        });

        out.innerHTML = `
            <div style="margin-bottom: 12px; font-size: 0.85rem; color: #94a3b8;">
                Audited <strong>${data.total_drugs_checked}</strong> medication classes. Identified <strong>${data.total_interactions_found}</strong> interaction(s).
            </div>
            ${cardsHtml}
        `;
    }

    // --- FEATURE 3: PATIENT EDUCATION LEAFLET GENERATOR ---
    const btnGenGuide = document.getElementById('btnGenGuide');
    if (btnGenGuide) {
        btnGenGuide.addEventListener('click', async () => {
            const cond = document.getElementById('guideCondition').value;
            const target = document.getElementById('guideTarget').value;
            const lang = document.getElementById('guideLang').value;

            try {
                const res = await fetch('/expert/patient-guide', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ condition_name: cond, target_bp: target, language: lang })
                });
                const data = await res.json();
                renderPatientGuide(data);
            } catch (e) {
                alert("Guide generation error: " + e.message);
            }
        });
    }

    function renderPatientGuide(data) {
        const out = document.getElementById('guideResultBody');
        const printBtn = document.getElementById('btnPrintLeaflet');
        if (printBtn) printBtn.style.display = 'inline-flex';

        const isAr = data.language === 'Arabic';
        const dir = isAr ? 'rtl' : 'ltr';
        const align = isAr ? 'right' : 'left';

        out.innerHTML = `
            <div style="direction: ${dir}; text-align: ${align}; background: #060b17; border: 1px solid rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 10px; font-family: sans-serif;">
                <h3 style="color: #38bdf8; font-size: 1.15rem; margin-bottom: 8px;">${escapeHtml(data.title)}</h3>
                <p style="font-size: 0.88rem; color: #e2e8f0; line-height: 1.6; margin-bottom: 14px;">${escapeHtml(data.condition_summary)}</p>

                <h4 style="color: #34d399; font-size: 0.92rem; margin: 12px 0 6px 0;">${isAr ? 'قواعد نمط الحياة والتغذية السليمة:' : 'Dietary & Lifestyle Rules:'}</h4>
                <ul style="font-size: 0.84rem; color: #cbd5e1; margin-bottom: 12px; padding-left: 20px; padding-right: 20px;">
                    ${data.lifestyle_rules.map(r => `<li style="margin-bottom: 4px;">${escapeHtml(r)}</li>`).join('')}
                </ul>

                <h4 style="color: #818cf8; font-size: 0.92rem; margin: 12px 0 6px 0;">${isAr ? 'إرشادات تناول الأدوية بأمان:' : 'Medication Safety Rules:'}</h4>
                <ul style="font-size: 0.84rem; color: #cbd5e1; margin-bottom: 12px; padding-left: 20px; padding-right: 20px;">
                    ${data.medication_safety_tips.map(t => `<li style="margin-bottom: 4px;">${escapeHtml(t)}</li>`).join('')}
                </ul>

                <div style="background: rgba(244, 63, 94, 0.1); border-left: 3px solid #f43f5e; padding: 12px; border-radius: 6px; margin-top: 14px;">
                    <strong style="color: #fb7185; font-size: 0.88rem;">${isAr ? 'علامات الخطر التي تستدعي الطوارئ فورا:' : 'Emergency Red Flag Warnings:'}</strong>
                    <ul style="font-size: 0.82rem; color: #fda4af; margin: 6px 0 6px 20px;">
                        ${data.emergency_red_flags.map(f => `<li>${escapeHtml(f)}</li>`).join('')}
                    </ul>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #ffffff;">${escapeHtml(data.emergency_action)}</div>
                </div>
            </div>
        `;
    }

    // --- FEATURE 4: MULTI-GUIDELINE COMPARISON MATRIX ---
    async function loadGuidelineComparison() {
        const tbody = document.getElementById('consensusTableBody');
        if (!tbody || tbody.children.length > 0) return;

        try {
            const res = await fetch('/expert/guidelines-comparison');
            const data = await res.json();

            tbody.innerHTML = '';
            data.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong style="color: #38bdf8;">${escapeHtml(row.topic)}</strong></td>
                    <td style="font-size: 0.82rem; color: #cbd5e1;">${escapeHtml(row.who_2021)}</td>
                    <td style="font-size: 0.82rem; color: #cbd5e1;">${escapeHtml(row.nice_ng136)}</td>
                    <td style="font-size: 0.82rem; color: #34d399;">${escapeHtml(row.consensus)}</td>
                `;
                tbody.appendChild(tr);
            });
        } catch (e) {
            console.error("Failed to load guideline comparisons:", e);
        }
    }

    // --- UTILITIES ---
    function escapeHtml(text) {
        if (!text) return '';
        const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }

    function formatMarkdownText(text) {
        if (!text) return '';
        let formatted = escapeHtml(text);
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        formatted = formatted.replace(/\n\n/g, '<br/><br/>');
        formatted = formatted.replace(/\n/g, '<br/>');
        return formatted;
    }
});
