import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram(output_path):
    # Set high quality styling
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 9.5
    
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis('off')
    
    # Palette: Professional IEEE Slate / Steel Blue / Muted Gold / Forest Green
    c_input = '#EBF3FA'
    c_attn = '#E3F2FD'
    c_router = '#FFF8E1'
    c_profile = '#F3E5F5'
    c_fusion = '#E8F5E9'
    c_triton = '#FBE9E7'
    c_output = '#ECEFF1'
    
    b_edge = '#37474F'
    
    # Helper to draw rounded boxes
    def draw_box(x, y, w, h, bg_color, title, subtitle=None, edge_color='#455A64', lw=1.2, radius=1.0):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius},rounding_size=1.5",
                                      facecolor=bg_color, edgecolor=edge_color, linewidth=lw)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontweight='bold', color='#212121', fontsize=9)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', color='#424242', fontsize=7.8)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontweight='bold', color='#212121', fontsize=8.8)

    # 1. INPUT
    draw_box(2, 21, 14, 10, c_input, "Input Prompt", "Tokens {x_0, ..., x_T-1}\nShape: (B, T, D)")

    # Arrow from Input to Model
    ax.annotate('', xy=(20, 26), xytext=(17, 26),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.5))
    
    # 2. TRANSFORMER BLOCK (dashed container)
    trans_rect = patches.FancyBboxPatch((20, 9), 29, 34, boxstyle="round,pad=1.2,rounding_size=2.0",
                                        facecolor='#FAFAFA', edgecolor='#78909C', linewidth=1.5, linestyle='--')
    ax.add_patch(trans_rect)
    ax.text(34.5, 41.5, "MoE Transformer Block (Layer l)", ha='center', fontweight='bold', color='#37474F', fontsize=9.5)

    # Branch A: Shared Self-Attention
    draw_box(22, 27, 25, 11, c_attn, "Shared Dense Attention", "Query-Key Projections\nAttention Mass A_hat(x_t)")

    # Branch B: Sparse FFN & Router
    draw_box(22, 12, 25, 11, c_router, "Sparse FFN & Router", "Router Gate: Top-K Experts\nSignature R_t = {E_t^(1)...E_t^(L)}")

    # Arrow splitting input to Branch A and Branch B
    ax.annotate('', xy=(22, 32.5), xytext=(18.5, 26),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.3, connectionstyle="angle,angleA=0,angleB=90,rad=3"))
    ax.annotate('', xy=(22, 17.5), xytext=(18.5, 26),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.3, connectionstyle="angle,angleA=0,angleB=-90,rad=3"))

    # 3. OFFLINE CALIBRATION / PROFILING
    draw_box(53, 3, 20, 9.5, c_profile, "Offline Expert Profiler", "Entropy H_bar, Route, Spec\nComputed once on D_calib")

    # Arrow from Router to Niche Score calculation
    draw_box(53, 15, 20, 11, c_router, "Routing Niche Scorer", "R(x_t) via Eq. (7)\nMin-Max Scaled R_hat in [0, 1]")
    ax.annotate('', xy=(53, 20.5), xytext=(48, 17.5),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.3))
    ax.annotate('', xy=(63, 15), xytext=(63, 13.5),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.2, linestyle=':'))

    # Arrow from Attention to Saliency
    ax.annotate('', xy=(53, 33), xytext=(48, 33),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.3))
    
    # 4. UNIFIED SALIENCY FUSION
    draw_box(53, 29, 20, 13, c_fusion, "Unified Saliency Fusion", 
             "S(x_t) = w_a A_hat + w_r R_hat\n+ w_s Sink + w_c Recency\nOrthogonal Signal (rho ≈ 0)")

    # Connect Routing Niche Scorer to Saliency Fusion
    ax.annotate('', xy=(63, 29), xytext=(63, 27),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.3))

    # 5. TOP-B SELECTION & SORTING
    draw_box(77, 29, 20, 11, c_triton, "Top-B Index Selection", "Select Top-(B-4) + 4 Sinks\nMonotonic sort: I_keep")
    ax.annotate('', xy=(77, 34.5), xytext=(74, 34.5),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.4))

    # 6. FUSED TRITON COMPACTION KERNEL
    draw_box(77, 13, 20, 12, c_triton, "Fused Triton Kernel", "In-SRAM Tile Compaction\n(B, H, T, D) -> (B, H, B, D)\nZero Mem Fragmentation")
    ax.annotate('', xy=(87, 25), xytext=(87, 29),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.4))

    # 7. OUTPUT COMPACTED CACHE
    draw_box(77, 1.5, 20, 8.5, c_output, "Compacted KV Cache", "2.02x - 2.05x Decode Speedup\nRetained Reasoning Anchors")
    ax.annotate('', xy=(87, 10.5), xytext=(87, 13),
                arrowprops=dict(arrowstyle="->", color=b_edge, lw=1.4))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Architecture diagram successfully saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram("paper_ieee/figures/ekva_architecture_diagram.png")
