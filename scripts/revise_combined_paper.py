"""Apply the requested editorial revision without running model evaluation."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / 'Docs/Combined Paper/Relative-Depth-Guided Residual Learning for Underwater Image Enhancement.md'

# All additions are IEEE journal/conference publications. Original method
# citations to U-Net (MICCAI) and Depth Anything V2 (NeurIPS) remain accurate.
ADDITIONS = {
    14: 'C. O. Ancuti, C. Ancuti, C. De Vleeschouwer, and P. Bekaert, “Color Balance and Fusion for Underwater Image Enhancement,” IEEE Trans. Image Process., vol. 27, no. 1, pp. 379-393, Jan. 2018, doi: 10.1109/TIP.2017.2759252. https://doi.org/10.1109/TIP.2017.2759252',
    15: 'C. Li, S. Anwar, J. Hou, R. Cong, C. Guo, and W. Ren, “Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding,” IEEE Trans. Image Process., vol. 30, pp. 4985-5000, 2021, doi: 10.1109/TIP.2021.3076367. https://doi.org/10.1109/TIP.2021.3076367',
    16: 'J. Li, K. A. Skinner, R. M. Eustice, and M. Johnson-Roberson, “WaterGAN: Unsupervised Generative Network to Enable Real-Time Color Correction of Monocular Underwater Images,” IEEE Robot. Autom. Lett., vol. 3, no. 1, pp. 387-394, Jan. 2018, doi: 10.1109/LRA.2017.2730363. https://doi.org/10.1109/LRA.2017.2730363',
    17: 'M. Badran and M. Torki, “DAUT: Underwater Image Enhancement Using Depth Aware U-shape Transformer,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2023, pp. 1830-1834, doi: 10.1109/ICIP49359.2023.10222895. https://doi.org/10.1109/ICIP49359.2023.10222895',
    18: 'K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html',
    19: 'L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 10371-10381. https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.html',
    20: 'H. Zhao, O. Gallo, I. Frosio, and J. Kautz, “Loss Functions for Image Restoration With Neural Networks,” IEEE Trans. Comput. Imaging, vol. 3, no. 1, pp. 47-57, Mar. 2017, doi: 10.1109/TCI.2016.2644865. https://doi.org/10.1109/TCI.2016.2644865',
    21: 'M. Yang and A. Sowmya, “An Underwater Color Image Quality Evaluation Metric,” IEEE Trans. Image Process., vol. 24, no. 12, pp. 6062-6071, Dec. 2015, doi: 10.1109/TIP.2015.2491020. https://doi.org/10.1109/TIP.2015.2491020',
}


FURTHER_ADDITIONS = {
    17: 'J. Y. Chiang and Y.-C. Chen, “Underwater Image Enhancement by Wavelength Compensation and Dehazing,” IEEE Trans. Image Process., vol. 21, no. 4, pp. 1756-1769, Apr. 2012, doi: 10.1109/TIP.2011.2179666. https://doi.org/10.1109/TIP.2011.2179666',
    18: 'P. L. J. Drews, E. R. Nascimento, S. S. C. Botelho, and M. F. M. Campos, “Underwater Depth Estimation and Image Restoration Based on Single Images,” IEEE Comput. Graph. Appl., vol. 36, no. 2, pp. 24-35, Mar.-Apr. 2016, doi: 10.1109/MCG.2016.26. https://doi.org/10.1109/MCG.2016.26',
    19: 'K. Panetta, C. Gao, and S. Agaian, “Human-Visual-System-Inspired Underwater Image Quality Measures,” IEEE J. Ocean. Eng., vol. 41, no. 3, pp. 541-551, Jul. 2016, doi: 10.1109/JOE.2015.2469915. https://doi.org/10.1109/JOE.2015.2469915',
    20: 'L. Peng, C. Zhu, and L. Bian, “U-Shape Transformer for Underwater Image Enhancement,” IEEE Trans. Image Process., vol. 32, pp. 3066-3079, 2023, doi: 10.1109/TIP.2023.3276332. https://doi.org/10.1109/TIP.2023.3276332',
    21: 'C. Fabbri, M. J. Islam, and J. Sattar, “Enhancing Underwater Imagery Using Generative Adversarial Networks,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2018, pp. 7159-7165, doi: 10.1109/ICRA.2018.8460552. https://doi.org/10.1109/ICRA.2018.8460552',
}


def extend_references(s: str) -> str:
    """Retain the academic/evidence split and add five directly relevant papers."""
    if '“Underwater Image Enhancement by Wavelength Compensation and Dehazing,”' in s:
        return s
    before, remainder = s.split('## References\n\n', 1)
    refs_text, appendix = remainder.split('## Appendix A.', 1)
    refs = {int(n): text.strip() for n, text in re.findall(r'(?ms)^\[(\d+)\] (.*?)(?=\n\n\[|\Z)', refs_text.strip())}
    assert set(refs) == set(range(1, 17))
    refs.update(FURTHER_ADDITIONS)
    inserts = {
        '### 2.2 Encoder-decoder reconstruction': 'Fabbri et al. investigate adversarial underwater image restoration and evaluate its effect on diver tracking [21]. This work connects enhancement with downstream visual perception and complements the later FUnIE-GAN formulation. The present experiments evaluate reconstruction fidelity and artifacts, without a downstream tracking experiment.',
        '### 2.3 Monocular relative depth': 'Peng et al. introduce a U-shape Transformer with channel-wise multiscale fusion and spatial global feature modeling, together with the LSUI dataset [20]. This provides a relevant alternative for modeling spatially varying attenuation and longer-range scene context. The present backbone uses convolutional encoder-decoder features and selective depth fusion.',
        '## 3. Methodology': 'Earlier model-based approaches also link scene depth to restoration. Chiang and Chen combine wavelength compensation with dehazing to address attenuation and scattering [17]. Drews et al. estimate scene depth and medium transmission from a single image using physical modeling and scene priors [18]. These approaches motivate depth-aware treatment of underwater degradation, whereas the present system supplies a normalized pretrained relative map to a learned residual network.',
        '### 5.2 Artifact assessment and checkpoint selection': 'Panetta et al. propose the underwater image quality measure (UIQM), combining colourfulness, sharpness and contrast components inspired by human visual perception [19]. UIQM offers additional context for underwater quality assessment alongside UCIQE and reference-based metrics. No UIQM measurements are reported for the current prototype.',
    }
    for anchor, paragraph in inserts.items():
        assert before.count(anchor) == 1
        before = before.replace(anchor, paragraph + '\n\n' + anchor)
    body = before + '## Appendix A.' + appendix
    order = list(dict.fromkeys(int(n) for n in re.findall(r'\[(\d+)\]', body)))
    assert set(order) == set(refs)
    numbers = {old: new for new, old in enumerate(order, 1)}
    body = re.sub(r'\[(\d+)\]', lambda m: f'[{numbers[int(m[1])]}]', body)
    before, appendix = body.split('## Appendix A.', 1)
    bibliography = '\n\n'.join(f'[{numbers[n]}] {refs[n]}' for n in order)
    return before + '## References\n\n' + bibliography + '\n\n## Appendix A.' + appendix


def revise_paper(s: str) -> str:
    if '### A.2 Project evidence sources' in s:
        return extend_references(s)
    s = re.sub(r'(?s)## 8\. Limitations\n.*?(?=## 9\. Conclusion)', '', s)
    s = s.replace('## 9. Conclusion', '## 8. Conclusion')
    s = s.replace('## 10. Evaluation Data Availability', '## 9. Evaluation Data Availability')
    s = s.replace('### 7.2 Depth sensitivity and artifact limitations', '### 7.2 Depth sensitivity and artifact assessment')
    s = s.replace('These limitations accompany the numerical results.', 'These observations accompany the numerical results.')

    def insert_before(anchor: str, content: str) -> None:
        nonlocal s
        assert s.count(anchor) == 1, anchor
        s = s.replace(anchor, content + '\n\n' + anchor)

    insert_before('### 2.2 Encoder-decoder reconstruction',
        'Colour correction and multiscale fusion provide a complementary approach to learned enhancement. Ancuti et al. combine colour compensation and white balancing with fusion of derived images to improve contrast and edge visibility [14]. Ucolor instead embeds RGB, HSV and Lab representations and guides its decoder with medium transmission [15]. These methods motivate consideration of both colour and spatial degradation; the present model uses input RGB statistics and learned relative-depth fusion rather than either published pipeline.')
    insert_before('### 2.3 Monocular relative depth',
        'Residual learning reformulates a prediction as a correction relative to its input [18]. Here this principle motivates the image-level residual output, while U-Net supplies the encoder-decoder organization. The implementation is a compact reconstruction network; it does not reproduce the classification architecture evaluated by He et al.')
    insert_before('### 2.4 Physics-aware recovery and prior depth conditioning',
        'The earlier Depth Anything model learns monocular depth representations by exploiting large-scale unlabeled imagery [19]. It provides context for pretrained depth guidance, while the implemented estimator remains Depth Anything V2 Small [5]. Neither citation supplies an underwater enhancement result for this implementation.')
    insert_before('## 3. Methodology',
        'WaterGAN uses in-air RGB and depth pairs to synthesize underwater training imagery for a colour-correction pipeline [16]. DAUT supplies estimated depth and underwater RGB to a depth-aware U-shape Transformer [17]. These are directly relevant precedents for connecting scene depth with enhancement, but their data-generation and transformer designs differ from the frozen-estimator residual CNN studied here. They establish related approaches, rather than matched experimental baselines for the reported results.')
    insert_before('## 4. Dataset and Training Protocol',
        'Zhao et al. study how pixel and structural loss choices affect neural image restoration [20]. Their analysis motivates examining reconstruction error together with structural agreement; it does not validate the particular weights or clipping policy used here, which are specified above.')
    insert_before('### 5.2 Artifact assessment and checkpoint selection',
        'For reference-free underwater evaluation, UCIQE combines chroma dispersion, luminance contrast and saturation [21]. It measures appearance attributes that complement reference-based PSNR and SSIM. UCIQE was not computed in the reported experiments, so it is discussed as related evaluation literature rather than an additional measured outcome.')

    before, remainder = s.split('## References\n\n', 1)
    refs_text, appendix = remainder.split('## Appendix A.', 1)
    refs = {int(n): text.strip() for n, text in re.findall(r'(?ms)^\[(\d+)\] (.*?)(?=\n\n\[|\Z)', refs_text.strip())}
    assert set(refs) == set(range(1, 14))
    records = {f'E{n - 8}': refs.pop(n) for n in range(9, 14)}
    refs.update(ADDITIONS)
    # Internal provenance stays identifiable and is never presented as IEEE work.
    body = before + '## Appendix A.' + appendix
    body = re.sub(r'\[(9|10|11|12|13)\]', lambda m: f'[E{int(m[1]) - 8}]', body)
    order = list(dict.fromkeys(int(n) for n in re.findall(r'\[(\d+)\]', body)))
    assert set(order) == set(refs), (order, refs.keys())
    numbers = {old: new for new, old in enumerate(order, 1)}
    body = re.sub(r'\[(\d+)\]', lambda m: f'[{numbers[int(m[1])]}]', body)
    before, appendix = body.split('## Appendix A.', 1)
    bibliography = '\n\n'.join(f'[{numbers[n]}] {refs[n]}' for n in order)
    body = before + '## References\n\n' + bibliography + '\n\n## Appendix A.' + appendix
    evidence = '### A.2 Project evidence sources\n\nThe following internal records identify the evidence for this prototype. They are separate from the scholarly bibliography and are not peer-reviewed IEEE publications.\n\n'
    evidence += '\n\n'.join(f'[{key}] {value}' for key, value in records.items()) + '\n\n'
    body = body.replace('## Appendix B.', evidence + '## Appendix B.')
    return extend_references(body)


if __name__ == '__main__':
    PAPER.write_text(revise_paper(PAPER.read_text()))
    subprocess.run([sys.executable, str(ROOT / 'scripts/render_combined_paper.py')], check=True)
