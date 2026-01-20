class LillithCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.render();
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          max-width: 20rem;
          width: 100%;
        }
        
        .card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 0.75rem;
          overflow: hidden;
          box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.04);
          transition: all 0.3s ease;
        }
        
        .card:hover {
          box-shadow: 0 20px 25px -5px rgba(0,0,0,0.08), 0 15px 15px -6px rgba(0,0,0,0.06);
        }
        
        .header {
          padding: 1rem;
          border-bottom: 1px solid #e5e7eb;
          background: linear-gradient(to right, #1d4ed8, #2563eb);
        }
        
        .header-label {
          font-size: 0.75rem;
          font-weight: 500;
          color: #bfdbfe;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        
        .user-info {
          display: flex;
          align-items: center;
          margin-top: 0.25rem;
        }
        
        .avatar {
          background: #dbeafe;
          color: #2563eb;
          border-radius: 50%;
          width: 2rem;
          height: 2rem;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-right: 0.5rem;
        }
        
        .username {
          font-size: 0.875rem;
          font-weight: 500;
          color: white;
          position: relative;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        
        .username::after {
          content: '';
          position: absolute;
          bottom: -2px;
          left: 0;
          width: 0;
          height: 1px;
          background: #2563eb;
          transition: all 0.3s ease;
        }
        
        .username:hover::after {
          width: 100%;
        }
        
        .menu {
          padding: 0.375rem 0;
        }
        
        .menu-item {
          position: relative;
          display: flex;
          align-items: center;
          padding: 0.625rem 1rem;
          font-size: 0.875rem;
          color: #374151;
          text-decoration: none;
          transition: all 0.2s ease;
        }
        
        .menu-item:hover {
          background: #eff6ff;
        }
        
        .menu-item.logout:hover {
          background: #fef2f2;
        }
        
        .menu-item::before {
          content: '';
          position: absolute;
          left: 0;
          top: 0;
          height: 100%;
          width: 4px;
          background: #3b82f6;
          border-radius: 0 4px 4px 0;
          opacity: 0;
          transform: scaleY(0.8);
          transition: all 0.2s ease;
        }
        
        .menu-item:hover::before {
          opacity: 1;
          transform: scaleY(1);
        }
        
        .menu-item.logout::before {
          background: #ef4444;
        }
        
        .icon-container {
          width: 2rem;
          height: 2rem;
          border-radius: 0.5rem;
          background: #dbeafe;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-right: 0.75rem;
          transition: all 0.2s ease;
        }
        
        .menu-item:hover .icon-container {
          background: #bfdbfe;
        }
        
        .menu-item.logout:hover .icon-container {
          background: #fecaca;
        }
        
        .icon {
          width: 1.25rem;
          height: 1.25rem;
          color: #2563eb;
          transition: all 0.2s ease;
        }
        
        .menu-item:hover .icon {
          color: #1d4ed8;
        }
        
        .menu-item.logout .icon {
          color: #ef4444;
        }
        
        .menu-item.logout:hover .icon {
          color: #dc2626;
        }
        
        .menu-text {
          font-weight: 500;
          flex: 1;
          transition: all 0.2s ease;
        }
        
        .menu-item:hover .menu-text {
          color: #1e3a8a;
        }
        
        .menu-item.logout:hover .menu-text {
          color: #dc2626;
        }
        
        .arrow {
          width: 0.75rem;
          height: 0.75rem;
          color: #9ca3af;
          margin-left: auto;
          transition: all 0.2s ease;
        }
        
        .menu-item:hover .arrow {
          color: #1d4ed8;
        }
        
        .menu-item.logout:hover .arrow {
          color: #ef4444;
        }
      </style>
      
      <div class="card">
        <div class="header">
          <p class="header-label">Signed in as</p>
          <div class="user-info">
            <div class="avatar">
              <svg fill="currentColor" viewBox="0 0 20 20" class="icon">
                <path clip-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" fill-rule="evenodd" />
              </svg>
            </div>
            <p class="username">john.parker@example.com</p>
          </div>
        </div>
        
        <div class="menu">
          <a href="#" class="menu-item">
            <div class="icon-container">
              <svg fill="currentColor" viewBox="0 0 20 20" class="icon">
                <path clip-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" fill-rule="evenodd" />
              </svg>
            </div>
            <span class="menu-text">Profile</span>
            <svg fill="currentColor" viewBox="0 0 20 20" class="arrow">
              <path clip-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" fill-rule="evenodd" />
            </svg>
          </a>
          
          <a href="#" class="menu-item">
            <div class="icon-container">
              <svg fill="currentColor" viewBox="0 0 20 20" class="icon">
                <path clip-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" fill-rule="evenodd" />
              </svg>
            </div>
            <span class="menu-text">Settings</span>
            <svg fill="currentColor" viewBox="0 0 20 20" class="arrow">
              <path clip-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" fill-rule="evenodd" />
            </svg>
          </a>
          
          <a href="#" class="menu-item logout">
            <div class="icon-container">
              <svg fill="currentColor" viewBox="0 0 20 20" class="icon">
                <path clip-rule="evenodd" d="M3 3a1 1 0 00-1 1v12a1 1 0 102 0V4a1 1 0 00-1-1zm10.293 9.293a1 1 0 001.414 1.414l3-3a1 1 0 000-1.414l-3-3a1 1 0 10-1.414 1.414L14.586 9H7a1 1 0 100 2h7.586l-1.293 1.293z" fill-rule="evenodd" />
              </svg>
            </div>
            <span class="menu-text">Logout</span>
            <svg fill="currentColor" viewBox="0 0 20 20" class="arrow">
              <path clip-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" fill-rule="evenodd" />
            </svg>
          </a>
        </div>
      </div>
    `;
  }
}

customElements.define('lillith-card', LillithCard);