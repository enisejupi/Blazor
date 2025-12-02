"""
PDF Report Generation
Gjenero raporte të detajuara në PDF për rezultatet e heqjes së zhurmës
"""
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os
from datetime import datetime
import cv2
import numpy as np
from io import BytesIO


class PDFReportGenerator:
    """Generate comprehensive PDF reports for denoising results"""
    
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Body style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
    
    def _image_to_reportlab(self, image_array, width=None, height=None):
        """Convert numpy array to ReportLab Image"""
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        # Encode to PNG
        _, buffer = cv2.imencode('.png', image_bgr)
        img_bytes = BytesIO(buffer.tobytes())
        
        # Create ReportLab Image
        img = Image(img_bytes, width=width, height=height)
        
        return img
    
    def generate_comparison_report(self, original, noisy, denoised_dict, metrics_dict, 
                                   noise_type, output_filename=None, language='sq'):
        """
        Generate comprehensive comparison report
        
        Args:
            original: Original clean image
            noisy: Noisy image
            denoised_dict: Dictionary of {method: denoised_image}
            metrics_dict: Dictionary of {method: metrics}
            noise_type: Type of noise applied
            output_filename: Output PDF filename
            language: 'sq' for Albanian, 'en' for English
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"denoising_report_{timestamp}.pdf"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                              rightMargin=50, leftMargin=50,
                              topMargin=50, bottomMargin=50)
        
        story = []
        
        # Title
        if language == 'sq':
            title_text = "Raporti i Krahasimit të Metodave të Heqjes së Zhurmës nga Imazhet"
            subtitle_text = "Image Denoising Methods Comparison Report"
        else:
            title_text = "Image Denoising Methods Comparison Report"
            subtitle_text = "Raporti i Krahasimit të Metodave"
        
        story.append(Paragraph(title_text, self.styles['CustomTitle']))
        story.append(Paragraph(subtitle_text, self.styles['Heading3']))
        story.append(Spacer(1, 20))
        
        # Date and time
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if language == 'sq':
            date_text = f"<b>Data e gjenerimit / Generation Date:</b> {current_time}"
        else:
            date_text = f"<b>Generation Date:</b> {current_time}"
        
        story.append(Paragraph(date_text, self.styles['CustomBody']))
        story.append(Spacer(1, 10))
        
        # Noise type
        if language == 'sq':
            noise_text = f"<b>Lloji i zhurmës / Noise Type:</b> {noise_type}"
        else:
            noise_text = f"<b>Noise Type:</b> {noise_type}"
        
        story.append(Paragraph(noise_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        if language == 'sq':
            story.append(Paragraph("Përmbledhje Ekzekutive / Executive Summary", 
                                 self.styles['CustomSubtitle']))
            summary_text = f"""
            Ky raport paraqet një krahasim gjithëpërfshirës të metodave të ndryshme të heqjes së zhurmës 
            nga imazhet. Janë testuar {len(denoised_dict)} metoda të ndryshme për heqjen e zhurmës së tipit 
            <b>{noise_type}</b>. Metodat përfshijnë qasje klasike (Median, Wiener, Wavelet) dhe metoda të 
            thella të të mësuarit (DnCNN), si dhe një qasje hibride që kombinon të dyja.
            """
        else:
            story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
            summary_text = f"""
            This report presents a comprehensive comparison of different image denoising methods. 
            {len(denoised_dict)} different methods were tested for removing <b>{noise_type}</b> noise. 
            The methods include classical approaches (Median, Wiener, Wavelet) and deep learning 
            methods (DnCNN), as well as a hybrid approach combining both.
            """
        
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Images section
        if language == 'sq':
            story.append(Paragraph("Krahasimi Vizual / Visual Comparison", 
                                 self.styles['CustomSubtitle']))
        else:
            story.append(Paragraph("Visual Comparison", self.styles['CustomSubtitle']))
        
        story.append(Spacer(1, 10))
        
        # Original and Noisy images
        img_width = 2.5 * inch
        img_height = 2.5 * inch
        
        images_table_data = []
        
        # First row: Original and Noisy
        row1 = [
            self._image_to_reportlab(original, width=img_width, height=img_height),
            self._image_to_reportlab(noisy, width=img_width, height=img_height)
        ]
        images_table_data.append(row1)
        
        if language == 'sq':
            labels_row1 = ["Imazhi Origjinal / Original", "Imazhi me Zhurmë / Noisy"]
        else:
            labels_row1 = ["Original Image", "Noisy Image"]
        
        images_table_data.append(labels_row1)
        
        # Create table
        img_table = Table(images_table_data, colWidths=[3*inch, 3*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, 1), 6),
        ]))
        
        story.append(img_table)
        story.append(Spacer(1, 20))
        
        # Denoised images (2x2 grid)
        denoised_images_data = []
        denoised_labels_data = []
        
        methods = list(denoised_dict.keys())
        for i in range(0, len(methods), 2):
            row_images = []
            row_labels = []
            
            for j in range(2):
                if i + j < len(methods):
                    method = methods[i + j]
                    row_images.append(
                        self._image_to_reportlab(denoised_dict[method], 
                                                width=img_width, height=img_height)
                    )
                    row_labels.append(method)
                else:
                    row_images.append("")
                    row_labels.append("")
            
            denoised_images_data.append(row_images)
            denoised_labels_data.append(row_labels)
        
        # Combine images and labels
        for img_row, label_row in zip(denoised_images_data, denoised_labels_data):
            story.append(Spacer(1, 10))
            
            table_data = [img_row, label_row]
            denoised_table = Table(table_data, colWidths=[3*inch, 3*inch])
            denoised_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, 1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 1), (-1, 1), 6),
            ]))
            
            story.append(denoised_table)
        
        story.append(PageBreak())
        
        # Metrics section
        if language == 'sq':
            story.append(Paragraph("Metrikat e Cilësisë / Quality Metrics", 
                                 self.styles['CustomSubtitle']))
            metrics_intro = """
            Metrikat e mëposhtme janë përdorur për të vlerësuar cilësinë e rezultateve të heqjes së zhurmës:
            <br/><br/>
            <b>• PSNR (Peak Signal-to-Noise Ratio):</b> Raporti i sinjalit maksimal ndaj zhurmës. 
            Vlera më e lartë tregon cilësi më të mirë (zakonisht > 30 dB është e mirë).<br/>
            <b>• SSIM (Structural Similarity Index):</b> Masa e ngjashmërisë strukturore. 
            Vlera më afër 1.0 tregon ngjashmëri më të madhe me origjinalin.<br/>
            <b>• MSE (Mean Squared Error):</b> Gabimi mesatar katror. Vlera më e ulët është më mirë.<br/>
            <b>• MAE (Mean Absolute Error):</b> Gabimi mesatar absolut. Vlera më e ulët është më mirë.
            """
        else:
            story.append(Paragraph("Quality Metrics", self.styles['CustomSubtitle']))
            metrics_intro = """
            The following metrics were used to evaluate the quality of denoising results:
            <br/><br/>
            <b>• PSNR (Peak Signal-to-Noise Ratio):</b> Ratio of maximum signal to noise. 
            Higher values indicate better quality (typically > 30 dB is good).<br/>
            <b>• SSIM (Structural Similarity Index):</b> Measure of structural similarity. 
            Values closer to 1.0 indicate greater similarity to the original.<br/>
            <b>• MSE (Mean Squared Error):</b> Mean squared error. Lower values are better.<br/>
            <b>• MAE (Mean Absolute Error):</b> Mean absolute error. Lower values are better.
            """
        
        story.append(Paragraph(metrics_intro, self.styles['CustomBody']))
        story.append(Spacer(1, 15))
        
        # Metrics table
        if language == 'sq':
            metrics_table_data = [['Metoda / Method', 'PSNR (dB)', 'SSIM', 'MSE', 'MAE']]
        else:
            metrics_table_data = [['Method', 'PSNR (dB)', 'SSIM', 'MSE', 'MAE']]
        
        for method in methods:
            metrics = metrics_dict[method]
            row = [
                method,
                f"{metrics['psnr']:.2f}",
                f"{metrics['ssim']:.4f}",
                f"{metrics['mse']:.2f}",
                f"{metrics['mae']:.2f}"
            ]
            metrics_table_data.append(row)
        
        metrics_table = Table(metrics_table_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Best method
        best_method = max(metrics_dict.items(), key=lambda x: x[1]['psnr'])[0]
        best_psnr = metrics_dict[best_method]['psnr']
        
        if language == 'sq':
            best_text = f"""
            <b>Metoda më e mirë bazuar në PSNR / Best Method based on PSNR:</b><br/>
            <font color='green' size='12'><b>{best_method}</b></font> me PSNR = <b>{best_psnr:.2f} dB</b>
            """
        else:
            best_text = f"""
            <b>Best Method based on PSNR:</b><br/>
            <font color='green' size='12'><b>{best_method}</b></font> with PSNR = <b>{best_psnr:.2f} dB</b>
            """
        
        story.append(Paragraph(best_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Conclusions
        if language == 'sq':
            story.append(Paragraph("Konkluzione / Conclusions", self.styles['CustomSubtitle']))
            conclusion_text = f"""
            Rezultatet tregojnë se për zhurmën e tipit <b>{noise_type}</b>, metoda <b>{best_method}</b> 
            ka dhënë rezultatin më të mirë me PSNR prej {best_psnr:.2f} dB. Metodat e ndryshme kanë 
            performanca të ndryshme në varësi të tipit të zhurmës dhe karakteristikave të imazhit.
            <br/><br/>
            Rekomandimet:<br/>
            • Për zhurmë Gaussian, metodat Wavelet dhe DnCNN japin rezultate të shkëlqyera.<br/>
            • Për zhurmë Salt & Pepper, filtri Median është më efektiv.<br/>
            • Për zhurmë Speckle, filtri Wiener është i rekomanduar.<br/>
            • Metoda hibride kombinon avantazhet e të dy qasjeve për rezultate optimale.
            """
        else:
            story.append(Paragraph("Conclusions", self.styles['CustomSubtitle']))
            conclusion_text = f"""
            The results show that for <b>{noise_type}</b> noise, the <b>{best_method}</b> method 
            performed best with a PSNR of {best_psnr:.2f} dB. Different methods have varying 
            performance depending on the noise type and image characteristics.
            <br/><br/>
            Recommendations:<br/>
            • For Gaussian noise, Wavelet and DnCNN methods provide excellent results.<br/>
            • For Salt & Pepper noise, Median filter is most effective.<br/>
            • For Speckle noise, Wiener filter is recommended.<br/>
            • Hybrid method combines advantages of both approaches for optimal results.
            """
        
        story.append(Paragraph(conclusion_text, self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        
        print(f"\n✅ Raporti u gjenerua me sukses / Report generated successfully:")
        print(f"   📄 {output_path}")
        
        return output_path


if __name__ == "__main__":
    # Test PDF generation
    print("Testing PDF report generation...")
    
    generator = PDFReportGenerator()
    
    # Create dummy data
    original = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    noisy = np.clip(original + np.random.randn(*original.shape) * 25, 0, 255).astype(np.uint8)
    
    denoised_dict = {
        'Median': np.clip(original + np.random.randn(*original.shape) * 10, 0, 255).astype(np.uint8),
        'Wavelet': np.clip(original + np.random.randn(*original.shape) * 8, 0, 255).astype(np.uint8),
        'DnCNN': np.clip(original + np.random.randn(*original.shape) * 5, 0, 255).astype(np.uint8),
    }
    
    metrics_dict = {
        'Median': {'psnr': 28.5, 'ssim': 0.85, 'mse': 145.2, 'mae': 8.3},
        'Wavelet': {'psnr': 30.2, 'ssim': 0.89, 'mse': 95.1, 'mae': 6.1},
        'DnCNN': {'psnr': 32.1, 'ssim': 0.92, 'mse': 61.5, 'mae': 4.2},
    }
    
    generator.generate_comparison_report(
        original, noisy, denoised_dict, metrics_dict,
        noise_type='Gaussian', language='sq'
    )
    
    print("PDF report generation test completed!")
