class LillithRadio extends HTMLElement {
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
          width: 100%;
          min-height: 100%;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        .fullscreen-container {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 100%;
          min-height: 100%;
          padding: 20px 0;
          box-sizing: border-box;
        }

        .radio-group-container {
          background-color: rgba(26, 27, 40, 0.7);
          padding: 20px 35px;
          border-radius: 20px;
          border: 1px solid rgba(255, 255, 255, 0.055);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          width: 90%;
          max-width: 280px;
          transition: all 0.3s ease;
        }

        .radio-label {
          display: flex;
          align-items: center;
          margin: 30px 0;
          cursor: pointer;
          position: relative;
          user-select: none;
        }

        .radio-input {
          display: none;
        }

        .radio-custom {
          width: 22px;
          height: 22px;
          background-color: transparent;
          border: 2px solid #5c5e79;
          border-radius: 50%;
          margin-right: 18px;
          position: relative;
          transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .radio-custom::before {
          content: "";
          position: absolute;
          width: 10px;
          height: 10px;
          background: #8a8b9f;
          border-radius: 50%;
          transform: scale(0);
          transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1);
        }

        .radio-custom::after {
          content: "";
          position: absolute;
          width: 34px;
          height: 34px;
          border: 2px solid transparent;
          border-radius: 50%;
          border-top-color: #00a6ff;
          opacity: 0;
          transform: scale(0.8);
          transition: all 0.4s ease;
        }

        .radio-text {
          font-size: 1.1rem;
          font-weight: 500;
          color: #c1c3d9;
          transition: color 0.3s ease;
        }

        .radio-label:hover .radio-input:not(:checked) + .radio-custom {
          transform: scale(1.1);
          border-color: #8a8daf;
        }

        .radio-label:hover .radio-text {
          color: #e2e4f4;
        }

        .radio-input:checked + .radio-custom {
          border-color: #00a6ff;
          transform: scale(0.9);
        }

        .radio-input:checked + .radio-custom::before {
          transform: scale(1);
          background-color: #00a6ff;
        }

        .radio-input:checked + .radio-custom::after {
          opacity: 1;
          transform: scale(1.3);
          animation: orbit 2.5s infinite linear;
          box-shadow: 0 0 30px #00a6ff, 0 0 80px rgba(0, 166, 255, 0.2);
        }

        .radio-input:checked ~ .radio-text {
          color: #ffffff;
          font-weight: 700;
        }

        .radio-label:nth-child(2) .radio-input:checked + .radio-custom {
          border-color: #e900ff;
          box-shadow: none;
        }
        .radio-label:nth-child(2) .radio-input:checked + .radio-custom::before {
          background-color: #e900ff;
        }
        .radio-label:nth-child(2) .radio-input:checked + .radio-custom::after {
          border-top-color: #e900ff;
          box-shadow: 0 0 30px #e900ff, 0 0 80px rgba(233, 0, 255, 0.2);
        }
        .radio-label:nth-child(2) .radio-input:checked ~ .radio-text {
          color: #ffffff;
        }

        .radio-label:nth-child(3) .radio-input:checked + .radio-custom {
          border-color: #00ffc2;
          box-shadow: none;
        }
        .radio-label:nth-child(3) .radio-input:checked + .radio-custom::before {
          background-color: #00ffc2;
        }
        .radio-label:nth-child(3) .radio-input:checked + .radio-custom::after {
          border-top-color: #00ffc2;
          box-shadow: 0 0 30px #00ffc2, 0 0 80px rgba(0, 255, 194, 0.2);
        }
        .radio-label:nth-child(3) .radio-input:checked ~ .radio-text {
          color: #ffffff;
        }

        @keyframes orbit {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }

        @media (max-width: 400px) {
          .radio-group-container {
            padding: 15px 25px;
          }

          .radio-label {
            margin: 25px 0;
          }

          .radio-text {
            font-size: 1rem;
          }
        }
      </style>
      
      <div class="fullscreen-container">
        <div class="radio-group-container">
          <label class="radio-label">
            <input checked type="radio" name="option" class="radio-input" />
            <span class="radio-custom"></span>
            <span class="radio-text">Helios Blue</span>
          </label>
          <label class="radio-label">
            <input type="radio" name="option" class="radio-input" />
            <span class="radio-custom"></span>
            <span class="radio-text">Cygnus Magenta</span>
          </label>
          <label class="radio-label">
            <input type="radio" name="option" class="radio-input" />
            <span class="radio-custom"></span>
            <span class="radio-text">Orion Lime</span>
          </label>
        </div>
      </div>
    `;
  }
}

customElements.define('lillith-radio', LillithRadio);