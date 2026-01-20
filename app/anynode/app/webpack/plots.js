// Fix plot display issues with gradio Tabs
function resetPlots() {
    document.querySelectorAll(".js-plotly-plot").forEach(plot => {
        Plotly.relayout(plot, { autosize: true });
    });
};
