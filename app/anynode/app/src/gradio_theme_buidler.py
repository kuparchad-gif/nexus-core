
#Theme builder for Gradio aplications
#Should export a JSON that looks like the following format
#
#import gradio as gr
#from gradio.themes.utils import colors, fonts, sizes
#
#custom_theme = gr.themes.Base(
#    primary_hue=colors.indigo,
#    secondary_hue=colors.purple,
#    neutral_hue=colors.zinc,
#    font=fonts.GoogleFont("Quicksand"),
#    font_mono=fonts.GoogleFont("JetBrains Mono"),
#    spacing_size=sizes.spacing_md,
#    radius_size=sizes.radius_lg,
#    text_size=sizes.text_md,
)

import gradio as gr
gr.themes.builder()