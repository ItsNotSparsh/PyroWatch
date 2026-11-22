"""
PDF Report Generator for PyroWatch

Generates comprehensive fire risk assessment reports in PDF format.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io
from PIL import Image
import numpy as np


def generate_fire_report(
    location_data,
    weather_data,
    risk_data,
    image_data=None,
    heatmap_data=None,
    hotspot_stats=None
):
    """
    Generate a comprehensive fire risk assessment PDF report.
    
    Args:
        location_data: dict with 'lat', 'lon', 'place_name' (optional)
        weather_data: dict with 'temp', 'humidity', 'wind_speed', 'wind_deg'
        risk_data: dict with 'risk_level', 'risk_score' (optional)
        image_data: PIL Image or numpy array (optional)
        heatmap_data: numpy array of attention heatmap (optional)
        hotspot_stats: dict with hotspot statistics (optional)
    
    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#FF6B35'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=6,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    
    # Title
    elements.append(Paragraph("🔥 PyroWatch Fire Risk Assessment Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
    elements.append(Paragraph(f"<i>Generated: {timestamp}</i>", normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Location Information
    elements.append(Paragraph("📍 Location Information", heading_style))
    location_table_data = [
        ['Latitude:', f"{location_data['lat']:.4f}°"],
        ['Longitude:', f"{location_data['lon']:.4f}°"],
    ]
    if 'place_name' in location_data and location_data['place_name']:
        location_table_data.insert(0, ['Location:', location_data['place_name']])
    
    location_table = Table(location_table_data, colWidths=[2*inch, 4*inch])
    location_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2937')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB'))
    ]))
    elements.append(location_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Weather Conditions
    elements.append(Paragraph("🌤️ Weather Conditions", heading_style))
    
    wind_dir_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    wind_dir_label = wind_dir_labels[int(((weather_data['wind_deg'] + 22.5) % 360) / 45)]
    
    weather_table_data = [
        ['Temperature:', f"{weather_data['temp']:.1f}°C"],
        ['Humidity:', f"{weather_data['humidity']:.1f}%"],
        ['Wind Speed:', f"{weather_data['wind_speed']:.1f} m/s"],
        ['Wind Direction:', f"{weather_data['wind_deg']:.0f}° ({wind_dir_label})"],
    ]
    
    weather_table = Table(weather_table_data, colWidths=[2*inch, 4*inch])
    weather_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2937')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB'))
    ]))
    elements.append(weather_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Risk Assessment
    elements.append(Paragraph("🎯 Risk Assessment", heading_style))
    
    risk_level = risk_data.get('risk_level', -1)
    if risk_level == 1:
        risk_text = "HIGH — Ignition Favorable"
        risk_color = colors.HexColor('#EF4444')
    elif risk_level == 0:
        risk_text = "LOW — Conditions Safe"
        risk_color = colors.HexColor('#10B981')
    else:
        risk_text = "NOT ASSESSED"
        risk_color = colors.HexColor('#6B7280')
    
    risk_table_data = [
        ['Risk Level:', risk_text],
    ]
    
    if 'risk_score' in risk_data:
        risk_table_data.append(['Risk Score:', f"{risk_data['risk_score']:.1f}/100"])
    
    risk_table = Table(risk_table_data, colWidths=[2*inch, 4*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2937')),
        ('TEXTCOLOR', (1, 0), (1, 0), risk_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB'))
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Image Analysis (if provided)
    if image_data is not None or heatmap_data is not None:
        elements.append(Paragraph("🔬 AI Image Analysis", heading_style))
        
        # Convert numpy arrays to PIL Images if needed
        if isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data)
        if isinstance(heatmap_data, np.ndarray):
            heatmap_data = Image.fromarray(heatmap_data)
        
        # Save images to BytesIO buffers
        img_list = []
        
        if image_data:
            img_buffer = io.BytesIO()
            image_data.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_list.append(('Original Image', img_buffer))
        
        if heatmap_data:
            heatmap_buffer = io.BytesIO()
            heatmap_data.save(heatmap_buffer, format='PNG')
            heatmap_buffer.seek(0)
            img_list.append(('Attention Heatmap', heatmap_buffer))
        
        # Add images side by side
        if len(img_list) == 2:
            img_table_data = [[
                RLImage(img_list[0][1], width=2.5*inch, height=2*inch),
                RLImage(img_list[1][1], width=2.5*inch, height=2*inch)
            ]]
            img_table = Table(img_table_data, colWidths=[3*inch, 3*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(img_table)
        elif len(img_list) == 1:
            elements.append(RLImage(img_list[0][1], width=4*inch, height=3*inch))
    
    # Hotspot Statistics (if provided)
    if hotspot_stats:
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("📊 Hotspot Statistics", heading_style))
        
        stats_table_data = [
            ['Total Hotspots:', f"{hotspot_stats.get('total', 0):,}"],
            ['Average per Day:', f"{hotspot_stats.get('avg_per_day', 0):.1f}"],
            ['Peak Day Count:', f"{hotspot_stats.get('peak_day', 0)}"],
            ['Date Range:', f"{hotspot_stats.get('date_range', 0)} days"],
        ]
        
        stats_table = Table(stats_table_data, colWidths=[2*inch, 4*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2937')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB'))
        ]))
        elements.append(stats_table)
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = """
    <para align=center>
    <font size=8 color="#6B7280">
    PyroWatch v2.0 — AI Wildfire Intelligence Platform<br/>
    Random Forest Risk Engine • Vision Transformer (ViT) Image Classifier • Wind-Driven Spread Simulator<br/>
    Data: NASA FIRMS • Weather: OpenWeatherMap
    </font>
    </para>
    """
    elements.append(Paragraph(footer_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer
