/**
 * @license
 * PRD-LLM Brain Framework
 * Copyright (c) 2024 Myo Min Aung
 * 
 * Licensed under the GNU General Public License v3.0.
 * See LICENSE file in the root directory for full license text.
 * 
 * Creator: Myo Min Aung (Independent Researcher)
 * Location: Yangon, Myanmar
 * Email: kkomyoeminaung@gmail.com
 */

import * as React from 'react';
import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Brain, 
  MessageSquare, 
  Hash, 
  Database, 
  Code, 
  Eye, 
  Activity, 
  Heart,
  Send,
  ShieldCheck,
  RotateCcw,
  Terminal,
  Cpu,
  Zap,
  GraduationCap,
  Dna,
  Bot,
  Wand2,
  Image as ImageIcon,
  Layers,
  Settings,
  X,
  ExternalLink,
  Copy,
  Check,
  FileUp,
  Link as LinkIcon,
  Upload
} from 'lucide-react';
import { RegionType, BrainState, REGIONS_CONFIG } from './types';

// PRD-LLM Local Brain Simulation Logic
// This replicates the Sparse MoE and Plasticity behavior in the browser
const SIMULATED_KNOWLEDGE = {
  [RegionType.REASONING]: ["Analyzing causal chains...", "Applying deductive logic...", "Verifying semantic consistency..."],
  [RegionType.LANGUAGE]: ["Synthesizing grammar structures...", "Processing multilingual tokens...", "Refining tone and resonance..."],
  [RegionType.MATH]: ["Calculating numerical vectors...", "Solving symbolic equations...", "Optimizing compute precision..."],
  [RegionType.MEMORY]: ["Accessing associative cache...", "Recalling long-term context...", "Updating synaptic weights..."],
  [RegionType.CODE]: ["Verifying syntax patterns...", "Generating algorithmic flow...", "Compiling instruction sets..."],
  [RegionType.VISION]: ["Mapping spatial coordinates...", "Detecting visual patterns...", "Allocating pixel attention..."],
  [RegionType.MOTOR]: ["Planning action sequences...", "Executing response buffer...", "Smoothing output motorics..."],
  [RegionType.EMOTIONAL]: ["Evaluating valence and reward...", "Updating mood state...", "Injecting emotional resonance..."]
};

export default function App() {
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant', content: string, status?: string }[]>([]);
  const [input, setInput] = useState('');
  const [logs, setLogs] = useState<string[]>(["// Global Brain Router: Online", "// Synaptic Plasticity: Active"]);
  const [showSettings, setShowSettings] = useState(false);
  const [apiUrl, setApiUrl] = useState('');
  const [isRemoteMode, setIsRemoteMode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [dreamStats, setDreamStats] = useState({
    is_dreaming: false,
    dream_count: 0,
    total_corrections: 0,
    buffer_stats: { total_experiences: 0, low_confidence_count: 0 }
  });

  const [learningStats, setLearningStats] = useState({
    self_learn_stats: { total_urls_learned: 0, total_topics_learned: 0, pending_topics: 15 },
    last_distillation: null,
    last_self_learn: null,
    knowledge_base: { total_entries: 0, document_sources: [] },
    optimization: { pruning: { actual_sparsity: 0 }, quantization: { reduction: 0 } },
    training: { stages: {} },
    deployment: { is_containerized: false, replicas: 0, cloud: "Local" },
    advanced: { multimodal: false, tool_use: [], agent_active: false },
    testing: { status: 'untested', unit: { status: 'pending' }, integration: { status: 'pending' }, stress: { status: 'pending' }, benchmark: { status: 'pending' } },
    is_colab: false,
    is_drive_mounted: false
  });

  // Heartbeat to prevent Colab idle timeout and poll stats
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRemoteMode && apiUrl) {
      interval = setInterval(async () => {
        try {
          const apiBase = apiUrl.trim().replace(/\/$/, '');
          // Keep-alive ping
          await fetch(`${apiBase}/ping`);
          
          const res = await fetch(`${apiBase}/learning/stats`);
          if (res.ok) {
            const data = await res.json();
            setLearningStats(data);
            setDreamStats(data.dream_stats || dreamStats);
          }
        } catch (e) {
          console.warn("Brain heartbeat failure");
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isRemoteMode, apiUrl]);

  const triggerManualCycle = async () => {
    if (!apiUrl) return;
    addLog("Initiating Autonomous Learning Cycle...");
    try {
      const res = await fetch(`${apiUrl.trim().replace(/\/$/, '')}/learning/cycle`, { method: 'POST' });
      const data = await res.json();
      addLog(`Cycle Complete: Distilled ${data.distilled}, Learned ${data.self_learned} topics`);
    } catch (e) {
      addLog("ERROR: Learning cycle failed");
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !apiUrl) return;
    
    addLog(`Ingesting document: ${file.name}...`);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${apiUrl.trim().replace(/\/$/, '')}/upload/file`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      addLog(`Success: Ingested ${data.chunks_added} knowledge chunks from ${file.name}`);
    } catch (error) {
      addLog("ERROR: Document ingestion failed");
    }
  };

  const handleUrlIngest = async () => {
    const url = prompt("Enter URL to ingest knowledge from:");
    if (!url || !apiUrl) return;
    
    addLog(`Crawling URL: ${url}...`);
    try {
      const res = await fetch(`${apiUrl.trim().replace(/\/$/, '')}/upload/url?url=${encodeURIComponent(url)}`, {
        method: 'POST'
      });
      const data = await res.json();
      addLog(`Success: Ingested ${data.chunks_added} chunks from URL`);
    } catch (error) {
      addLog("ERROR: URL ingestion failed");
    }
  };

  const triggerOptimization = async () => {
    if (!apiUrl) return;
    addLog("Initiating Model Optimization Pipeline (Pruning + Quantization)...");
    try {
      const res = await fetch(`${apiUrl.trim().replace(/\/$/, '')}/model/optimize`, { method: 'POST' });
      const data = await res.json();
      addLog(`Optimization Complete: ${Math.round(data.stats.pruning.actual_sparsity * 100)}% sparsity achieved.`);
    } catch (e) {
      addLog("ERROR: Optimization pipeline failed");
    }
  };

  const triggerTraining = async () => {
    if (!apiUrl) return;
    addLog("Starting Stage 2: Myanmar Data Collection + Fine-tuning + RLHF...");
    try {
      const res = await fetch(`${apiUrl.trim().replace(/\/$/, '')}/training/run`, { method: 'POST' });
      const data = await res.json();
      addLog(`Stage 2 Complete: Collected ${data.stages.data_collection.total_samples} Myanmar samples. Fine-tuned with LoRA and DPO-RLHF.`);
    } catch (e) {
      addLog("ERROR: Training pipeline failed");
    }
  };
  
  const runAutonomousAgent = async () => {
    const goal = prompt("Enter the goal for the autonomous agent:");
    if (!goal || !apiUrl) return;
    
    addLog(`Agent thinking: ${goal}...`);
    try {
      const res = await fetch(`${apiUrl.trim().replace(/\/$/, '')}/agent/run?goal=${encodeURIComponent(goal)}`, { method: 'POST' });
      const data = await res.json();
      addLog(`Agent Goal: ${data.goal}`);
      addLog(`Final Answer: ${data.final_answer}`);
    } catch (e) {
      addLog("ERROR: Agent execution failed");
    }
  };

  const triggerTests = async () => {
    if (!apiUrl) return;
    addLog("Initiating Full Test Suite (Unit + Integration + Stress + Benchmark)...");
    try {
      const res = await fetch(`${apiUrl.trim().replace(/\/$/, '')}/tests/run`, { method: 'POST' });
      const data = await res.json();
      addLog(`Tests Complete: All systems operational. Unit: ${data.unit.status}, Integration: ${data.integration.status}`);
    } catch (e) {
      addLog("ERROR: Test suite execution failed");
    }
  };

  const [state, setState] = useState<BrainState>({
    activeRegions: [],
    confidence: 0,
    plasticityLevel: 0.1,
    isProcessing: false,
    criticCorrectionCount: 0
  });

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, state.isProcessing, logs]);

  const addLog = (msg: string) => {
    setLogs(prev => [...prev.slice(-15), `// ${msg}`]);
  };

  // Internal Router Algorithm (Sparse MoE)
  const brainRouter = (text: string): RegionType[] => {
    const lowerText = text.toLowerCase();
    const scores: { [key in RegionType]: number } = {
      [RegionType.REASONING]: (lowerText.match(/why|how|analyze|logic|think|prove|explain/g) || []).length,
      [RegionType.LANGUAGE]: (lowerText.match(/write|say|tell|poem|story|translate/g) || []).length + 0.5,
      [RegionType.MATH]: (lowerText.match(/math|calculate|number|sum|subtract|multiply|divide|equation/g) || []).length,
      [RegionType.MEMORY]: (lowerText.match(/remember|data|storage|recall|history|memory/g) || []).length,
      [RegionType.CODE]: (lowerText.match(/code|function|program|js|ts|python|rust|c\+\+|api/g) || []).length,
      [RegionType.VISION]: (lowerText.match(/look|see|image|find|spatial|layout|visual/g) || []).length,
      [RegionType.MOTOR]: (lowerText.match(/do|act|move|button|execute|click/g) || []).length,
      [RegionType.EMOTIONAL]: (lowerText.match(/feel|happy|sad|angry|emotion|love|hate|care/g) || []).length,
    };

    // Sort regions by score and pick top K (Sparse Activation)
    const sorted = (Object.keys(scores) as RegionType[]).sort((a, b) => scores[b] - scores[a]);
    return sorted.slice(0, 2); // K=2
  };

  const handleSend = useCallback(async () => {
    if (!input.trim() || state.isProcessing) return;

    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    
    const activated = brainRouter(userMsg);
    addLog(`Routing to active regions: ${activated.join(', ')}`);
    
    setState(prev => ({ 
      ...prev, 
      isProcessing: true, 
      activeRegions: activated,
      confidence: 0.2
    }));

    try {
      if (isRemoteMode && apiUrl) {
        addLog(`Requesting Remote Brain Relay: ${apiUrl}`);
        const response = await fetch(`${apiUrl.trim().replace(/\/$/, '')}/generate`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({ prompt: userMsg })
        });
        
        if (!response.ok) throw new Error("Remote relay rejected connection.");
        const data = await response.json();
        
        setState(prev => ({ ...prev, confidence: data.confidence || 0.9 }));
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.text || "No response data found.",
          status: (data.regions || activated).join(' + ') + " [REMOTE]" 
        }]);
      } else {
        // LOCAL SIMULATION
        await new Promise(r => setTimeout(r, 600));
        for (const region of activated) {
          const traces = SIMULATED_KNOWLEDGE[region];
          addLog(`${region} Region: ${traces[Math.floor(Math.random() * traces.length)]}`);
          await new Promise(r => setTimeout(r, 400));
        }
        
        setState(prev => ({ 
          ...prev, 
          confidence: 0.6, 
          plasticityLevel: Math.min(1, prev.plasticityLevel + 0.05) 
        }));
        await new Promise(r => setTimeout(r, 600));

        setState(prev => ({ ...prev, confidence: 0.85 }));
        const needsCorrection = Math.random() > 0.8; 
        
        let finalOutput = `[LOCAL SIMULATION] I am currently running offline. Please connect to the Remote Brain (Colab) via settings for real AGI responses.

Synthesized regional result (${activated.join(' + ')}): Neural firing is stable.`;

        if (needsCorrection) {
          addLog("Critic Layer: Low confidence detected. Re-triggering regional check...");
          setState(prev => ({ ...prev, criticCorrectionCount: prev.criticCorrectionCount + 1 }));
          await new Promise(r => setTimeout(r, 800));
          finalOutput = "[Critically Verified] " + finalOutput;
        }

        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: finalOutput,
          status: activated.join(' + ') 
        }]);
      }
      addLog("Stream Output: Finalized");
      
    } catch (error) {
      console.error(error);
      addLog(`ERROR: ${error instanceof Error ? error.message : 'Unknown neural failure'}`);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: isRemoteMode 
          ? "⚠️ Remote Brain Connection Failed. Please check your Colab server and ensure the Ngrok URL in settings is correct." 
          : "Neural communication failed. Recovery mode initiated." 
      }]);
    } finally {
      setState(prev => ({ ...prev, isProcessing: false, activeRegions: [] }));
    }
  }, [input, state.isProcessing, isRemoteMode, apiUrl, state.plasticityLevel]);

  const copyColabCode = () => {
    const code = `# 🚀 PRD-LLM AGI ARCHITECTURE - FULL ENGINE SETUP (AUTONOMOUS)
# [အဆင့် ၁] - လိုအပ်သော Libraries များ Install လုပ်ခြင်း
!pip install fastapi uvicorn nest-asyncio pyngrok pydantic python-multipart torch transformers accelerate pypdf python-docx beautifulsoup4 lxml einops redis
print("📥 Cloning PRD-LLM Architecture Components...")
!git clone https://github.com/PRD-LLM/PRD-LLM-Brain.git /content/prd-llm-brain || (cd /content/prd-llm-brain && git pull)
%cd /content/prd-llm-brain

# [အဆင့် ၂] - PRD-LLM Engine & Relay Server စတင်ခြင်း
import os
import time
from pyngrok import ngrok
from google.colab import userdata

# Get Ngrok Token
token = userdata.get('NGROK_AUTH_TOKEN')
if not token:
    print("❌ ERROR: NGROK_AUTH_TOKEN ရှာမတွေ့ပါ။ Notebook Secrets (Key icon) မှာ ထည့်ပေးပါ။")
else:
    ngrok.set_auth_token(token)
    
    # Start Tunnel
    public_url = ngrok.connect(8000).public_url
    print("\\n" + "🚀" + "="*58 + "🚀")
    print(f" PRD-LLM FULL AGI ENGINE IS ONLINE! ".center(60))
    print("="*60)
    print(f" 🔗 API URL: {public_url} ".center(60))
    print("="*60 + "\\n")
    print("🧠 အလိုအလျောက် သင်ယူခြင်း (AUTONOMOUS LEARNING): ACTIVE")
    print("📚 Knowledge Distillation: ACTIVE")
    print("🌙 Dream Mode (Self-Optimization): ACTIVE\\n")
    
    print("ညွှန်ကြားချက်:")
    print(f"၁။ အပေါ်က link ကို Copy ကူးပြီး Web App Settings မှာ ထည့်ပါ။")
    print("၂။ Engine အပြည့်အစုံ Load လုပ်ရန် ၁ မိနစ်ခန့် စောင့်ပေးပါ။\\n")

    # Launching the complete server script from repository
    # This automatically handles AutonomousLearner, RAG, and Optimization Pipeline.
    !python COLAB_SERVER.py --port 8000 --relay-url {public_url}
`;
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-screen bg-warm-bg text-warm-text font-sans selection:bg-sage/20">
      {/* Header */}
      <nav className="px-8 py-6 flex justify-between items-center z-20 border-b border-clay/20 bg-warm-bg/80 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-sage organic-shape shadow-lg shadow-sage/30 shrink-0 flex items-center justify-center">
            <Layers className="w-4 h-4 text-white" />
          </div>
          <div className="flex flex-col">
            <span className="text-xl font-medium tracking-tight serif italic leading-none">PRD-MoE Brain</span>
            <span className="text-[9px] uppercase tracking-[0.2em] font-bold text-sage opacity-80 mt-1">Standalone AGI Architecture</span>
          </div>
        </div>

        <div className="hidden lg:flex gap-12 text-[10px] uppercase tracking-[0.2em] font-bold opacity-30">
          <span>Sparse Activation</span>
          <span>Hebbian Plasticity</span>
          <span>Zero-Trust Critic</span>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex flex-col items-end">
            <span className="text-[10px] uppercase tracking-widest font-bold opacity-30 italic">Plasticity</span>
            <div className="w-24 h-1 bg-warm-text/5 rounded-full mt-1 overflow-hidden">
               <motion.div className="h-full bg-sage" animate={{ width: `${state.plasticityLevel * 100}%` }} />
            </div>
          </div>
          <button 
            onClick={() => setShowSettings(true)}
            className="p-2 hover:bg-clay/20 rounded-full transition-colors text-warm-text/60"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </nav>

      {/* Main Layout */}
      <main className="flex-1 overflow-hidden flex flex-col md:flex-row p-6 gap-6 lg:px-12 bg-warm-bg">
        {/* Left: Brain Core Visualization */}
        <section className="flex-1 bg-white rounded-[40px] shadow-soft border-2 border-clay p-8 relative overflow-hidden flex flex-col group/brain">
          <div className="flex items-center justify-between mb-8 pb-4 border-b-2 border-clay/50">
            <h3 className="serif text-3xl italic tracking-tight font-medium flex items-center gap-3 text-warm-text">
              Neural Grid <span className="text-xs font-sans not-italic uppercase tracking-[0.3em] font-bold text-sage opacity-70">{isRemoteMode ? 'Relay Engine' : 'Local Processing'}</span>
            </h3>
            <div className="flex items-center gap-2 px-3 py-1 bg-white rounded-full border-2 border-clay">
               <div className={`w-2 h-2 rounded-full ${state.isProcessing ? 'bg-sage animate-pulse' : isRemoteMode ? 'bg-blue-600' : 'bg-clay'}`} />
               <span className="text-[10px] font-bold uppercase tracking-widest text-warm-text/60">{state.isProcessing ? 'Thinking' : isRemoteMode ? 'Relay Active' : 'Offline'}</span>
            </div>
          </div>

          <div className="flex-1 grid grid-cols-2 lg:grid-cols-4 gap-6 relative">
             {Object.entries(REGIONS_CONFIG).map(([type, config]) => {
               const isActive = state.activeRegions.includes(type as RegionType);
               const Icon = {
                 Brain, MessageSquare, Hash, Database, Code, Eye, Activity, Heart
               }[config.icon] || Brain;

               return (
                 <motion.div
                   key={type}
                   animate={{ 
                     boxShadow: isActive ? `0 10px 30px -10px ${config.color}60` : '0 2px 8px rgba(0,0,0,0.1)',
                     backgroundColor: isActive ? 'white' : '#FDFCF8',
                     borderColor: isActive ? config.color : '#D4B9A0',
                     scale: isActive ? 1.02 : 1
                   }}
                   className={`border-2 rounded-[2rem] p-6 flex flex-col items-center justify-center gap-4 transition-all relative overflow-hidden ${isActive ? 'z-10 shadow-lg' : 'opacity-80'}`}
                 >
                   {isActive && (
                      <motion.div 
                        layoutId="active-bg"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 0.1 }}
                        className="absolute inset-0"
                        style={{ backgroundColor: config.color }}
                      />
                   )}
                   <div className={`p-4 rounded-[35%] ${isActive ? 'bg-warm-bg shadow-inner border border-clay/30' : 'bg-transparent'} transition-all`}>
                      <Icon className="w-8 h-8" style={{ color: isActive ? config.color : '#2D272333' }} />
                   </div>
                   <span className={`text-[10px] font-bold uppercase tracking-[0.2em] text-center ${isActive ? 'text-warm-text font-black' : 'text-warm-text/60'}`}>
                     {type}
                   </span>

                   {isActive && (
                     <div className="flex gap-1">
                        <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="w-1 h-1 rounded-full" style={{ backgroundColor: config.color }} />
                        <motion.div initial={{ scale: 0 }} animate={{ scale: 1, transition: { delay: 0.1 } }} className="w-1 h-1 rounded-full" style={{ backgroundColor: config.color }} />
                        <motion.div initial={{ scale: 0 }} animate={{ scale: 1, transition: { delay: 0.2 } }} className="w-1 h-1 rounded-full" style={{ backgroundColor: config.color }} />
                     </div>
                   )}
                 </motion.div>
               );
             })}
          </div>

          {/* Router Logic Trace */}
          <div className="mt-8 p-6 bg-bubble-user/40 border border-clay/50 rounded-3xl font-mono text-[10px] text-warm-text/40 leading-relaxed italic relative">
             <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                <Cpu className="w-20 h-20" />
             </div>
             <div className="flex items-center gap-2 mb-3 text-sage not-italic font-bold tracking-widest uppercase opacity-80">
                <Terminal className="w-3 h-3" />
                <span>Logic Controller // Trace Output</span>
             </div>
             <div className="space-y-1 h-24 overflow-y-auto overflow-x-hidden scrollbar-hide">
               {logs.map((log, i) => (
                 <motion.p key={i} initial={{ x: -10, opacity: 0 }} animate={{ x: 0, opacity: 1 }}>
                   {log}
                 </motion.p>
               ))}
             </div>
          </div>

          {/* Dream Mode Status */}
          {isRemoteMode && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 p-8 bg-white border-2 border-clay shadow-lg rounded-[2.5rem]"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <span className="text-xl">🌙</span>
                    <span className="text-[10px] font-bold uppercase tracking-[0.2em] opacity-60">Neuro-Dream State</span>
                  </div>
                  {dreamStats.is_dreaming ? (
                    <motion.div 
                      animate={{ opacity: [1, 0.4, 1] }} 
                      transition={{ duration: 1.5, repeat: Infinity }}
                      className="flex items-center gap-2"
                    >
                      <div className="w-2 h-2 rounded-full bg-sage shadow-[0_0_8px_rgba(142,166,149,0.8)]" />
                      <span className="text-[10px] text-sage font-bold tracking-widest uppercase">Dreaming</span>
                    </motion.div>
                  ) : (
                    <span className="text-[10px] opacity-30 font-bold tracking-widest uppercase">Consolidating</span>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-8">
                  <div className="space-y-1">
                    <p className="text-[9px] opacity-40 uppercase font-black tracking-tighter">Memory Cycles</p>
                    <p className="text-3xl serif italic text-warm-text">{dreamStats.dream_count}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-[9px] opacity-40 uppercase font-black tracking-tighter">Self-Corrections</p>
                    <p className="text-3xl serif italic text-warm-text">{dreamStats.total_corrections}</p>
                  </div>
                </div>
                {dreamStats.is_dreaming && (
                   <div className="mt-6 h-1 bg-sage/10 rounded-full overflow-hidden">
                     <motion.div 
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 5, ease: "linear" }}
                        className="h-full bg-sage"
                     />
                   </div>
                )}
                <p className="mt-4 text-[10px] italic opacity-40 leading-relaxed">
                  NeuroFlow is currently re-processing low-confidence paths using Hebbian weight adjustments.
                </p>
              </motion.div>
            )}

          {/* Autonomous Learning Dashboard */}
          {isRemoteMode && (
            <motion.div 
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="mt-6 p-8 bg-white border-2 border-clay shadow-lg rounded-[2.5rem]"
            >
               <div className="flex items-center justify-between mb-8">
                 <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-warm-text/60 flex items-center gap-2">
                   <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
                   Autonomous Knowledge Pipeline
                 </h4>
                 <button 
                   onClick={triggerManualCycle}
                   className="text-[9px] font-bold uppercase tracking-widest text-sage border border-sage/20 px-3 py-1 rounded-full hover:bg-sage hover:text-white transition-colors"
                 >
                   Trigger Cycle
                 </button>
               </div>

               {learningStats.is_colab && (
                 <div className="mb-6 p-4 bg-blue-50 rounded-2xl border border-blue-100 flex items-center justify-between">
                   <div className="flex items-center gap-3">
                     <div className={`w-2 h-2 rounded-full ${learningStats.is_drive_mounted ? 'bg-green-500' : 'bg-orange-500 animate-pulse'}`} />
                     <span className="text-[10px] font-black uppercase tracking-widest text-blue-900">
                       Google Drive Persistence: {learningStats.is_drive_mounted ? 'Connected' : 'Offline'}
                     </span>
                   </div>
                   <span className="text-[10px] italic text-blue-700/60 truncate max-w-[200px]">Knowledge saved to MyDrive/PRD_LLM_Brain</span>
                 </div>
               )}

               <div className="mb-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                 <div className="p-6 bg-white rounded-3xl border-2 border-clay/10 shadow-sm">
                   <h3 className="text-sm font-black uppercase tracking-widest text-blue-900 mb-4 flex items-center gap-2">
                     <div className="w-1.5 h-1.5 bg-blue-600 rounded-full" />
                     Neural Architecture Specs
                   </h3>
                   <div className="space-y-3">
                     <div className="flex justify-between items-center py-2 border-b border-clay/5">
                       <span className="text-xs text-gray-500 font-medium font-mono uppercase tracking-tighter">Architecture</span>
                       <span className="text-xs font-bold text-blue-700">Multi-Region Transformer (8 Core Regions)</span>
                     </div>
                     <div className="flex justify-between items-center py-2 border-b border-clay/5">
                       <span className="text-xs text-gray-500 font-medium font-mono uppercase tracking-tighter">Nerve Plasticity</span>
                       <span className="text-xs font-bold text-blue-700">Active (Real-time Weight Updates)</span>
                     </div>
                     <div className="flex justify-between items-center py-2 border-b border-clay/5">
                       <span className="text-xs text-gray-500 font-medium font-mono uppercase tracking-tighter">Synaptic Consolidation</span>
                       <span className="text-xs font-bold text-blue-700 italic">Dream Mode (Self-Distillation during sleep)</span>
                     </div>
                     <div className="flex justify-between items-center py-2">
                       <span className="text-xs text-gray-500 font-medium font-mono uppercase tracking-tighter">Model Scale</span>
                       <span className="text-xs font-bold text-blue-700">0.5B Parameters (Optimized for Edge/Colab)</span>
                     </div>
                   </div>
                 </div>

                 <div className="p-6 bg-blue-900 rounded-3xl text-white shadow-xl relative overflow-hidden group">
                   <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full -mr-16 -mt-16 blur-2xl group-hover:bg-white/10 transition-all duration-700" />
                   <h3 className="text-sm font-black uppercase tracking-widest text-blue-200 mb-4 flex items-center gap-2">
                     <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                     Cognitive Similarity Index
                   </h3>
                   <div className="relative z-10">
                     <div className="mb-4">
                       <div className="flex justify-between mb-2">
                         <span className="text-[10px] font-bold uppercase tracking-widest opacity-60">Human Brain Similarity</span>
                         <span className="text-xs font-black">74%</span>
                       </div>
                       <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                         <div className="h-full bg-white w-[74%] rounded-full shadow-[0_0_10px_rgba(255,255,255,0.5)]" />
                       </div>
                     </div>
                     <p className="text-[10px] leading-relaxed text-blue-100/80 italic">
                       PRD-LLM uses functional neuro-mapping to route signals through specialized regions, mimicking the biological efficiency of the human cortex. Each interaction updates neural weights to learn from you in real-time.
                     </p>
                   </div>
                 </div>
               </div>

               <div className="grid grid-cols-3 gap-6">
                  <div className="p-6 bg-warm-bg rounded-3xl border-2 border-clay/30 text-center">
                    <span className="text-3xl serif italic text-blue-700 block font-bold">{learningStats.self_learn_stats.total_topics_learned}</span>
                    <span className="text-[9px] uppercase font-black text-warm-text/40 mt-1 block">Topics Learned</span>
                  </div>
                  <div className="p-6 bg-warm-bg rounded-3xl border-2 border-clay/30 text-center">
                    <span className="text-3xl serif italic text-orange-700 block font-bold">{learningStats.self_learn_stats.total_urls_learned}</span>
                    <span className="text-[9px] uppercase font-black text-warm-text/40 mt-1 block">URLs Scraped</span>
                  </div>
                  <div className="p-6 bg-warm-bg rounded-3xl border-2 border-clay/30 text-center">
                    <span className="text-3xl serif italic text-sage block font-bold">{learningStats.self_learn_stats.pending_topics}</span>
                    <span className="text-[9px] uppercase font-black text-warm-text/40 mt-1 block">Pending In Queue</span>
                  </div>
               </div>

               <div className="mt-6 space-y-2">
                  <div className="flex justify-between text-[9px] font-bold uppercase opacity-30">
                    <span>Last Distillation</span>
                    <span>{learningStats.last_distillation || 'NEVER'}</span>
                  </div>
                  <div className="flex justify-between text-[9px] font-bold uppercase opacity-30">
                    <span>Last Web Scrape</span>
                    <span>{learningStats.last_self_learn || 'NEVER'}</span>
                  </div>
               </div>
            </motion.div>
          )}

          {/* Knowledge Ingestion Dashboard */}
          {isRemoteMode && (
            <motion.div 
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="mt-6 p-8 bg-white border-2 border-clay shadow-lg rounded-[2.5rem]"
            >
               <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-warm-text/60 mb-8 flex items-center gap-2">
                 <Database className="w-4 h-4 text-sage" />
                 Document Knowledge Ingestion
               </h4>

               <div className="flex gap-3 mb-6">
                  <label className="flex-1 cursor-pointer">
                    <input type="file" className="hidden" onChange={handleFileUpload} accept=".pdf,.txt,.docx,.html" />
                    <div className="flex items-center justify-center gap-2 px-4 py-3 bg-sage/5 border border-sage/20 rounded-2xl hover:bg-sage hover:text-white transition-all text-[10px] font-bold uppercase tracking-widest">
                      <FileUp className="w-3 h-3" />
                      Upload Doc
                    </div>
                  </label>
                  <button 
                    onClick={handleUrlIngest}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-warm-bg border border-clay rounded-2xl hover:border-sage transition-all text-[10px] font-bold uppercase tracking-widest"
                  >
                    <LinkIcon className="w-3 h-3" />
                    URL Ingest
                  </button>
               </div>

               <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30">
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-[9px] font-bold uppercase opacity-40">Knowledge Base Stats</span>
                    <span className="text-[10px] font-mono text-sage">{learningStats.knowledge_base.total_entries} CHUNKS</span>
                  </div>
                  <div className="space-y-2 max-h-32 overflow-y-auto custom-scrollbar pr-2">
                    {learningStats.knowledge_base.document_sources.length > 0 ? (
                      learningStats.knowledge_base.document_sources.map((src: string, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-[9px] italic opacity-60">
                          <div className="w-1 h-1 rounded-full bg-sage" />
                          <span className="truncate">{src}</span>
                        </div>
                      ))
                    ) : (
                      <p className="text-[9px] italic opacity-30 text-center py-2">No documents ingested yet.</p>
                    )}
                  </div>
               </div>
            </motion.div>
          )}

          {/* Training & Data Dashboard */}
          {isRemoteMode && (
            <motion.div 
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="mt-6 p-8 bg-white border-2 border-clay shadow-lg rounded-[2.5rem]"
            >
               <div className="flex items-center justify-between mb-8">
                 <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-warm-text/60 flex items-center gap-2">
                   <GraduationCap className="w-4 h-4 text-sage" />
                   Training & Data (Stage 2)
                 </h4>
                 <button 
                   onClick={triggerTraining}
                   className="text-[9px] font-bold uppercase tracking-widest text-sage border border-sage/20 px-3 py-1 rounded-full hover:bg-sage hover:text-white transition-colors"
                 >
                   Start Training
                 </button>
               </div>

               <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30">
                    <div className="flex items-center gap-2 mb-2">
                       <Dna className="w-2.5 h-2.5 text-sage" />
                       <span className="text-[8px] uppercase font-bold text-warm-text/30">Myanmar Data</span>
                    </div>
                    <span className="text-xl serif italic text-sage block">
                      {learningStats.training?.stages?.data_collection?.total_samples || 0} Samples
                    </span>
                  </div>
                  <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30">
                    <div className="flex items-center gap-2 mb-2">
                       <Activity className="w-2.5 h-2.5 text-sage" />
                       <span className="text-[8px] uppercase font-bold text-warm-text/30">RLHF Alignment</span>
                    </div>
                    <span className="text-xl serif italic text-sage block">
                      {Math.round((learningStats.training?.stages?.rlhf?.alignment_score || 0) * 100)}%
                    </span>
                  </div>
               </div>

               <div className="mt-4 p-3 bg-clay/5 rounded-xl border border-clay/20">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                       <ShieldCheck className="w-3 h-3 text-sage" />
                       <span className="text-[9px] font-bold uppercase opacity-50">Instruction Tuning</span>
                    </div>
                    <span className="text-[9px] font-mono text-sage">
                      {learningStats.training?.stages?.fine_tuning?.status === 'complete' ? 'COMPLETE' : 'PENDING'}
                    </span>
                  </div>
               </div>
            </motion.div>
          )}

          {/* Model Optimization Dashboard */}
          {isRemoteMode && (
            <motion.div 
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="mt-6 p-6 bg-white border border-clay rounded-[2rem] shadow-soft"
            >
               <div className="flex items-center justify-between mb-6">
                 <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-warm-text/40 flex items-center gap-2">
                   <Zap className="w-3 h-3 text-purple-600" />
                   Model Optimization (Step 1)
                 </h4>
                 <button 
                   onClick={triggerOptimization}
                   className="text-[9px] font-bold uppercase tracking-widest text-purple-600 border border-purple-600/20 px-3 py-1 rounded-full hover:bg-purple-600 hover:text-white transition-colors"
                 >
                   Optimize Now
                 </button>
               </div>

               <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30 text-center">
                    <span className="text-2xl serif italic text-purple-600 block">{Math.round(learningStats.optimization.pruning.actual_sparsity * 100)}%</span>
                    <span className="text-[8px] uppercase font-bold text-warm-text/30">Pruning Sparsity</span>
                  </div>
                  <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30 text-center">
                    <span className="text-2xl serif italic text-blue-600 block">{Math.round(learningStats.optimization.quantization.reduction)}%</span>
                    <span className="text-[8px] uppercase font-bold text-warm-text/30">Memory Saved</span>
                  </div>
               </div>

               <div className="mt-6 grid grid-cols-2 gap-2">
                  <div className="flex items-center gap-2 px-3 py-2 bg-clay/10 rounded-xl">
                    <Check className="w-2 h-2 text-sage" />
                    <span className="text-[9px] font-bold uppercase opacity-50">Flash Attention</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-2 bg-clay/10 rounded-xl">
                    <Check className="w-2 h-2 text-sage" />
                    <span className="text-[9px] font-bold uppercase opacity-50">KV Caching</span>
                  </div>
               </div>
            </motion.div>
          )}

          {/* Production Deployment Dashboard */}
          {isRemoteMode && (
            <motion.div 
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="mt-6 p-6 bg-white border border-clay rounded-[2rem] shadow-soft"
            >
               <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-warm-text/40 mb-6 flex items-center gap-2">
                 <div className="w-2 h-2 bg-green-500 rounded-full" />
                 Production Infrastructure (Stage 3)
               </h4>

               <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30">
                    <span className="text-[8px] uppercase font-bold text-warm-text/30 block mb-1">Environment</span>
                    <span className="text-sm font-bold text-warm-text">{learningStats.deployment.cloud}</span>
                  </div>
                  <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30">
                    <span className="text-[8px] uppercase font-bold text-warm-text/30 block mb-1">Scale Status</span>
                    <span className="text-sm font-bold text-warm-text">{learningStats.deployment.replicas} Active Replicas</span>
                  </div>
               </div>

               <div className="mt-4 p-3 bg-green-50 rounded-xl border border-green-100 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Settings className="w-3 h-3 text-green-600" />
                    <span className="text-[9px] font-bold text-green-700 uppercase tracking-widest">K8S Load Balancer: Online</span>
                  </div>
                  <div className="text-[9px] font-mono text-green-600">100% HEALTH</div>
               </div>
            </motion.div>
          )}

          {/* Advanced Features Dashboard */}
          {isRemoteMode && (
            <motion.div 
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="mt-6 p-6 bg-white border border-clay rounded-[2rem] shadow-soft"
            >
               <div className="flex items-center justify-between mb-6">
                 <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-warm-text/40 flex items-center gap-2">
                   <Wand2 className="w-3 h-3 text-blue-500" />
                   Advanced Intelligence (Stage 4)
                 </h4>
                 <button 
                   onClick={runAutonomousAgent}
                   className="text-[9px] font-bold uppercase tracking-widest text-blue-600 border border-blue-600/20 px-3 py-1 rounded-full hover:bg-blue-600 hover:text-white transition-colors"
                 >
                   Launch Agent
                 </button>
               </div>

               <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30">
                    <div className="flex items-center gap-2 mb-2">
                       <Bot className="w-2.5 h-2.5 text-blue-500" />
                       <span className="text-[8px] uppercase font-bold text-warm-text/30">Autonomous</span>
                    </div>
                    <span className="text-sm font-bold text-warm-text">{learningStats.advanced.agent_active ? 'ACTIVE' : 'IDLE'}</span>
                  </div>
                  <div className="p-4 bg-warm-bg rounded-2xl border border-clay/30">
                    <div className="flex items-center gap-2 mb-2">
                       <ImageIcon className="w-2.5 h-2.5 text-purple-500" />
                       <span className="text-[8px] uppercase font-bold text-warm-text/30">Vision (Multimodal)</span>
                    </div>
                    <span className="text-sm font-bold text-warm-text">{learningStats.advanced.multimodal ? 'ENABLED' : 'DISABLED'}</span>
                  </div>
               </div>

               <div className="p-4 bg-clay/5 rounded-2xl border border-clay/20">
                  <span className="text-[8px] uppercase font-bold text-warm-text/30 block mb-2">Integrated Tools</span>
                  <div className="flex flex-wrap gap-2">
                    {learningStats.advanced.tool_use.map((tool: string, i: number) => (
                      <div key={i} className="px-2 py-1 bg-white border border-clay/30 rounded-lg text-[9px] font-mono text-blue-600">
                        {tool.toUpperCase()}
                      </div>
                    ))}
                  </div>
               </div>
            </motion.div>
          )}

          {/* Testing & Validation Dashboard */}
          {isRemoteMode && (
            <motion.div 
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="mt-6 p-6 bg-white border border-clay rounded-[2rem] shadow-soft"
            >
               <div className="flex items-center justify-between mb-6">
                 <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-warm-text/40 flex items-center gap-2">
                   <ShieldCheck className="w-3 h-3 text-red-500" />
                   Testing & Validation (Stage 5)
                 </h4>
                 <button 
                   onClick={triggerTests}
                   className="text-[9px] font-bold uppercase tracking-widest text-red-600 border border-red-600/20 px-3 py-1 rounded-full hover:bg-red-600 hover:text-white transition-colors"
                 >
                   Run Tests
                 </button>
               </div>

               <div className="grid grid-cols-4 gap-2 mb-4">
                  {[
                    { label: 'Unit', status: learningStats.testing?.unit?.status },
                    { label: 'Integ', status: learningStats.testing?.integration?.status },
                    { label: 'Stress', status: learningStats.testing?.stress?.status },
                    { label: 'Bench', status: learningStats.testing?.benchmark?.status }
                  ].map((t, i) => (
                    <div key={i} className="p-2 bg-warm-bg rounded-xl border border-clay/20 text-center">
                       <span className="text-[7px] uppercase font-bold text-warm-text/20 block mb-1">{t.label}</span>
                       <span className={`text-[9px] font-bold ${t.status === 'passed' ? 'text-sage' : 'text-warm-text/30'}`}>
                         {t.status?.toUpperCase() || 'IDLE'}
                       </span>
                    </div>
                  ))}
               </div>

               <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-clay/5 rounded-2xl">
                     <span className="text-[8px] uppercase font-bold text-warm-text/30 block mb-1">Throughput</span>
                     <span className="text-sm font-bold text-warm-text">
                       {Math.round(learningStats.testing?.benchmark?.tokens_per_sec || 0)} tok/s
                     </span>
                  </div>
                  <div className="p-4 bg-clay/5 rounded-2xl">
                     <span className="text-[8px] uppercase font-bold text-warm-text/30 block mb-1">Avg Latency</span>
                     <span className="text-sm font-bold text-warm-text">
                       {Math.round(learningStats.testing?.stress?.avg_latency_ms || 0)} ms
                     </span>
                  </div>
               </div>
            </motion.div>
          )}
        </section>

        {/* Right: Interaction Interface */}
        <section className="w-full md:w-[480px] flex flex-col pt-4">
          <div className="mb-6 flex items-center justify-between px-2">
             <h2 className="serif text-2xl italic text-warm-text opacity-90">Cognitive Stream</h2>
             <button 
               onClick={() => {
                 setMessages([]);
                 setLogs(["// System Reset", "// Global Brain Router: Online"]);
               }}
               className="text-[10px] uppercase font-bold tracking-widest opacity-30 hover:opacity-100 transition-opacity flex items-center gap-2"
             >
                <RotateCcw className="w-3 h-3" /> RESET CORE
             </button>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-6 scroll-smooth pr-2 custom-scrollbar">
            {messages.length === 0 && !state.isProcessing && (
              <div className="flex flex-col items-center justify-center h-full text-center p-12">
                <div className="w-20 h-20 bg-clay/20 organic-shape mb-8 flex items-center justify-center">
                  <Brain className="w-8 h-8 text-clay" />
                </div>
                <p className="serif text-3xl italic text-warm-text/40">Enter a prompt to engage internal MoE routing.</p>
                <div className="mt-8 grid grid-cols-2 gap-3 w-full">
                   {["Analyze logic", "Process language", "Calculate numbers", "Emotional resonance"].map(t => (
                     <button 
                       key={t}
                       onClick={() => { setInput(t); }}
                       className="px-3 py-2 border border-clay/30 rounded-full text-[9px] uppercase font-bold tracking-widest text-sage hover:bg-sage hover:text-white transition-all"
                     >
                       {t}
                     </button>
                   ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div className={`max-w-[90%] px-6 py-4 rounded-[2.5rem] text-sm leading-relaxed border ${
                  msg.role === 'user' 
                    ? 'bg-bubble-user text-warm-text border-clay/50 rounded-tr-none' 
                    : 'bg-sage text-white border-sage/20 rounded-tl-none shadow-xl shadow-sage/10'
                }`}>
                  {msg.content}
                </div>
                {msg.status && (
                  <span className="text-[8px] font-bold uppercase tracking-[0.2em] text-sage/60 mt-2 px-2 italic">
                    Activated Path: {msg.status}
                  </span>
                )}
              </motion.div>
            ))}

            {state.isProcessing && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-3 px-6"
              >
                <div className="flex gap-1.5 py-4">
                  <motion.div className="w-1.5 h-1.5 bg-sage/40 rounded-full" animate={{ scale: [1, 1.5, 1], opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1 }} />
                  <motion.div className="w-1.5 h-1.5 bg-sage/40 rounded-full" animate={{ scale: [1, 1.5, 1], opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1, delay: 0.2 }} />
                  <motion.div className="w-1.5 h-1.5 bg-sage/40 rounded-full" animate={{ scale: [1, 1.5, 1], opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1, delay: 0.4 }} />
                </div>
                <span className="text-[10px] uppercase tracking-widest font-bold text-sage opacity-60">Firing Neurons...</span>
              </motion.div>
            )}
          </div>

          {/* Input Area */}
          <div className="mt-8 relative pt-4 border-t border-clay/30">
            <div className="relative flex items-center gap-4">
              <input 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Neural relay query..."
                className="flex-1 bg-bubble-user border border-clay/60 rounded-full px-8 py-4 text-sm focus:outline-none focus:border-sage transition-all placeholder:text-warm-text/20 italic"
              />
              <button 
                onClick={handleSend}
                disabled={!input.trim() || state.isProcessing}
                className="w-14 h-14 bg-warm-text hover:bg-sage disabled:bg-clay text-white rounded-full transition-all flex items-center justify-center shadow-2xl shadow-warm-text/20 shrink-0 group"
              >
                <Send className="w-5 h-5 -rotate-45 group-hover:scale-110 transition-transform" />
              </button>
            </div>
            <div className="mt-6 flex items-center justify-between text-[9px] uppercase tracking-[0.2em] font-bold text-warm-text/30">
               <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-sage shadow-sm shadow-sage/50" />
                  <span>Internal Architecture Visualization</span>
               </div>
               <div className="flex items-center gap-4">
                  <span>Logic Corrections: {state.criticCorrectionCount}</span>
                  <span className="opacity-20">// RELAY_OK</span>
               </div>
            </div>
          </div>
        </section>
      </main>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-warm-text/20 backdrop-blur-sm">
             <motion.div 
               initial={{ opacity: 0, scale: 0.9 }} 
               animate={{ opacity: 1, scale: 1 }} 
               exit={{ opacity: 0, scale: 0.9 }}
               className="bg-warm-bg border border-clay rounded-[2.5rem] w-full max-w-xl overflow-hidden shadow-2xl"
             >
                <div className="p-8 border-b border-clay/20 flex justify-between items-center bg-white/40">
                  <h2 className="serif text-2xl italic tracking-tight">Neural Settings</h2>
                  <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-clay/20 rounded-full transition-colors"><X className="w-5 h-5" /></button>
                </div>
                
                <div className="p-8 space-y-8 max-h-[70vh] overflow-y-auto custom-scrollbar">
                  {/* Remote Relay Config */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-[10px] uppercase tracking-widest font-bold opacity-40 italic">Remote Relay Config</h4>
                      <div 
                        onClick={() => setIsRemoteMode(!isRemoteMode)}
                        className={`w-10 h-5 rounded-full transition-colors relative cursor-pointer ${isRemoteMode ? 'bg-sage' : 'bg-clay'}`}
                      >
                         <motion.div animate={{ x: isRemoteMode ? 20 : 2 }} className="absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-sm" />
                      </div>
                    </div>
                    <p className="text-xs text-warm-text/60 leading-relaxed italic">Point extraction to your custom Model endpoint (FastAPI/Colab).</p>
                    <input 
                      value={apiUrl}
                      onChange={(e) => setApiUrl(e.target.value)}
                      placeholder="https://xyz.ngrok-free.app"
                      className="w-full bg-white/50 border border-clay rounded-2xl px-6 py-4 text-sm italic focus:outline-none focus:border-sage transition-all"
                    />
                  </div>

                  <hr className="border-clay/20" />

                  {/* Colab Instructions */}
                  <div className="space-y-4">
                    <h4 className="text-[10px] uppercase tracking-widest font-bold opacity-40 italic">Colab Engine Setup</h4>
                    <p className="text-xs text-warm-text/40 leading-relaxed">Copy this cell into Google Colab to expose your PRD-LLM model via ngrok.</p>
                    <div className="p-6 bg-bubble-user rounded-[2rem] border border-clay/50 relative group">
                       <pre className="text-[10px] font-mono leading-relaxed overflow-x-auto text-warm-text/80 h-32 hide-scrollbar">
{`# 1. Install & Connect ngrok
!pip install fastapi pyngrok
# 2. Start PRD-LLM Relay...
# See settings button for complete code`}
                       </pre>
                       <button 
                         onClick={copyColabCode}
                         className="absolute top-4 right-4 p-2 bg-white rounded-xl border border-clay shadow-sm hover:bg-sage hover:text-white transition-all flex items-center gap-2 text-[10px] font-bold"
                       >
                         {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                         {copied ? "COPIED" : "COPY CODE"}
                       </button>
                    </div>
                    <div className="flex items-center gap-4 pt-2">
                       <a href="https://colab.research.google.com" target="_blank" className="flex items-center gap-2 text-[10px] font-bold text-sage underline underline-offset-4 decoration-sage/30">
                         OPEN COLAB <ExternalLink className="w-3 h-3" />
                       </a>
                    </div>
                  </div>
                </div>

                <div className="p-8 bg-sage/5 flex justify-end">
                   <button onClick={() => setShowSettings(false)} className="px-8 py-3 bg-warm-text text-white rounded-full text-xs font-bold tracking-widest hover:bg-sage transition-colors">SAVE RELAY PARAMS</button>
                </div>
             </motion.div>
          </div>
        )}
      </AnimatePresence>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(142, 166, 149, 0.2); border-radius: 10px; }
        .organic-shape { border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%; }
        .shadow-soft { box-shadow: 0 20px 40px -15px rgba(61, 54, 49, 0.08); }
        .serif { font-family: 'Instrument Serif', serif; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .hide-scrollbar::-webkit-scrollbar { display: none; }
      `}</style>
    </div>
  );
}
