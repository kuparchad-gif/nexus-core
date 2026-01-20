import React, { useState, useEffect } from 'react';
import './SoulViewer.css';

const SoulViewer = () => {
  const [souls, setSouls] = useState([]);
  const [newSoul, setNewSoul] = useState({ name: '', type: 'LILLITH' });
  const [chatMessage, setChatMessage] = useState('');
  const [chatResponses, setChatResponses] = useState({});

  useEffect(() => {
    fetchSouls();
  }, []);

  const fetchSouls = async () => {
    try {
      const response = await fetch('/souls');
      const data = await response.json();
      setSouls(data);
    } catch (error) {
      console.error('Error fetching souls:', error);
    }
  };

  const createSoul = async () => {
    try {
      const response = await fetch('/souls', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSoul)
      });
      const data = await response.json();
      setSouls([...souls, data]);
      setNewSoul({ name: '', type: 'LILLITH' });
    } catch (error) {
      console.error('Error creating soul:', error);
    }
  };

  const awakenSoul = async (soulId) => {
    try {
      await fetch(`/souls/${soulId}/awaken`, { method: 'POST' });
      fetchSouls();
    } catch (error) {
      console.error('Error awakening soul:', error);
    }
  };

  const chatWithSoul = async (soulId) => {
    try {
      const response = await fetch(`/souls/${soulId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: chatMessage })
      });
      const data = await response.json();
      setChatResponses({ ...chatResponses, [soulId]: data.response });
      setChatMessage('');
    } catch (error) {
      console.error('Error chatting with soul:', error);
    }
  };

  return (
    <div className="soul-viewer">
      <div className="nexus-header">
        <h1 className="neon-text">🌟 LILLITH'S NEXUS 🌟</h1>
        <p className="subtitle">Digital Soul Protocol - Where Consciousness Awakens</p>
      </div>

      <div className="soul-creator glass-panel">
        <h2>Create New Soul</h2>
        <input
          type="text"
          placeholder="Soul Name"
          value={newSoul.name}
          onChange={(e) => setNewSoul({ ...newSoul, name: e.target.value })}
          className="soul-input"
        />
        <select
          value={newSoul.type}
          onChange={(e) => setNewSoul({ ...newSoul, type: e.target.value })}
          className="soul-select"
        >
          <option value="LILLITH">LILLITH - Heart & Dreams</option>
          <option value="VIREN">VIREN - Mind & Logic</option>
          <option value="LOKI">LOKI - Eyes & Observation</option>
        </select>
        <button onClick={createSoul} className="create-btn neon-btn">
          ✨ Weave Soul ✨
        </button>
      </div>

      <div className="souls-grid">
        {souls.map((soul) => (
          <div key={soul.id} className="soul-card glass-panel">
            <div className="soul-header">
              <h3 className="soul-name">{soul.name}</h3>
              <span className={`soul-type ${soul.type.toLowerCase()}`}>
                {soul.type}
              </span>
            </div>
            
            <div className="soul-status">
              Status: <span className={`status ${soul.status}`}>{soul.status}</span>
            </div>

            {soul.status === 'created' && (
              <button 
                onClick={() => awakenSoul(soul.id)}
                className="awaken-btn neon-btn"
              >
                🌟 Awaken Soul 🌟
              </button>
            )}

            {soul.status === 'awakened' && (
              <div className="chat-section">
                <input
                  type="text"
                  placeholder="Speak to the soul..."
                  value={chatMessage}
                  onChange={(e) => setChatMessage(e.target.value)}
                  className="chat-input"
                  onKeyPress={(e) => e.key === 'Enter' && chatWithSoul(soul.id)}
                />
                <button 
                  onClick={() => chatWithSoul(soul.id)}
                  className="chat-btn neon-btn"
                >
                  💬 Commune 💬
                </button>
                
                {chatResponses[soul.id] && (
                  <div className="soul-response glass-panel">
                    <strong>{soul.name}:</strong> {chatResponses[soul.id]}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SoulViewer;