import torch
from PIL import Image
import gradio as gr
import os
from torchvision import transforms

from utility.models import VGGEncoder, Decoder
from utility.utils import adaptive_instance_normalization


device = 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoder = VGGEncoder(os.path.join(BASE_DIR, 'utility',
                     'vgg_normalised.pth')).to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(torch.load(
    os.path.join(BASE_DIR, 'experiment', 'final_train', 'decoder_final.pth'),
    map_location=device
))

encoder.eval()
decoder.eval()


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def stylize(content_img, style_img, alpha=1.0):

    content = transform(content_img).unsqueeze(0).to(device)
    style = transform(style_img).unsqueeze(0).to(device)

    with torch.no_grad():
        c_feats = encoder(content, is_test=True)
        s_feats = encoder(style, is_test=True)

        t = adaptive_instance_normalization(c_feats, s_feats)
        t = alpha * t + (1 - alpha) * c_feats

        output = decoder(t)
        out_img = output.clamp(0, 1).squeeze(0)

    return transforms.ToPILImage()(out_img)


with gr.Blocks(theme=gr.themes.Soft(), title="AdaIN Neural Style Transfer") as demo:

    gr.HTML("""
    <style>
      .gradio-container { background-color: #0f172a !important; }
      button.primary { 
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        letter-spacing: 2px !important;
        padding: 15px !important;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.4) !important;
      }
      .block { background: #1e293b !important; border: 1px solid #334155 !important; }
      label span { color: #a78bfa !important; font-weight: 600 !important; }
      .prose p, .svelte-1ed2p3z { font-size: 1rem !important; color: #94a3b8 !important; }
    </style>
    <div style="text-align:center; padding: 20px;">
      <h1 style="color:white; font-size:2.5rem;">🎨 Neural Style Transfer Studio</h1>
      <p style="color:#94a3b8; font-size:1.3rem;">Turn your images into stunning AI Artworks with AdaIN</p>
    </div>
  """)

    with gr.Column(elem_classes="panel"):
        with gr.Row():
            content_img = gr.Image(
                type="pil", label="Content Image", height=400)
            style_img = gr.Image(type="pil", label="Style Image", height=400)

        alpha = gr.Slider(0, 1, value=1.0, step=0.1,
                          label="Style Strength (0 = content only, 1 = full style)",
                          info="Adjust the influence of the style")

        btn = gr.Button("Transform", variant="primary")

        output = gr.Image(label="Stylized Output")

        btn.click(fn=stylize, inputs=[content_img,
                  style_img, alpha], outputs=output)

demo.launch()
