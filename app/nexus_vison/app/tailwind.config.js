/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        silver: "#C0C0C0",
        purple: "#7E57C2",
        labwhite: "#F7F7FB"
      },
      borderRadius: {
        '2xl': '1rem'
      }
    },
  },
  plugins: [],
}
