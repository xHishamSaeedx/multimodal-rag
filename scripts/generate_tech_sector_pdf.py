#!/usr/bin/env python3
"""
Simple script to generate a 5-page PDF on the tech sector and its growth trends.
Includes tables but no graphs.

Usage:
    python generate_tech_sector_pdf.py [output_path]

Requirements:
    pip install reportlab
"""

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER


def create_tech_sector_pdf(output_path: str = "tech_sector_report.pdf"):
    """Generate a 5-page PDF on the tech sector with tables."""
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    # Container for PDF content
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#283593'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    normal_style = styles['Normal']
    normal_style.alignment = TA_JUSTIFY
    
    # Page 1: Title and Introduction
    story.append(Paragraph("Technology Sector: Growth and Trends", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "The technology sector continues to be one of the most dynamic and rapidly evolving "
        "industries in the global economy. Over the past decade, technological innovation has "
        "transformed how we work, communicate, and conduct business. This report examines the "
        "current state of the tech sector, its growth trajectory, emerging trends, and key "
        "factors driving its expansion.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The sector encompasses a wide range of sub-industries including software development, "
        "cloud computing, artificial intelligence, cybersecurity, e-commerce, and telecommunications. "
        "Each of these areas has experienced significant growth, driven by increasing digitalization "
        "of businesses and consumer adoption of new technologies. The convergence of these technologies "
        "has created unprecedented opportunities for innovation and disruption across traditional "
        "industries.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Digital transformation has become a strategic imperative for organizations worldwide. "
        "Companies that were once hesitant to adopt new technologies are now investing heavily "
        "in digital infrastructure to remain competitive. This shift has been accelerated by "
        "the COVID-19 pandemic, which forced many businesses to rapidly implement remote work "
        "solutions and digital customer engagement platforms.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The technology sector's influence extends far beyond its own boundaries. Traditional "
        "industries such as finance, healthcare, manufacturing, and retail are increasingly "
        "relying on technology to enhance efficiency, improve customer experiences, and create "
        "new revenue streams. This cross-industry integration has become a significant driver "
        "of tech sector growth.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Investment in research and development within the technology sector remains at historically "
        "high levels. Major technology companies allocate substantial portions of their revenues "
        "to R&D activities, driving continuous innovation. This commitment to innovation ensures "
        "that the sector remains at the cutting edge of technological advancement.",
        normal_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Table 1: Global Tech Sector Revenue Growth
    story.append(Paragraph("Global Tech Sector Revenue Growth (2020-2024)", heading_style))
    
    table1_data = [
        ['Year', 'Revenue<br/>(Trillion USD)', 'Growth<br/>Rate (%)', 'Key Drivers'],
        ['2020', '4.2', '5.3', 'Remote work, digital transformation'],
        ['2021', '4.8', '14.3', 'Cloud migration, e-commerce'],
        ['2022', '5.3', '10.4', 'AI/ML, cybersecurity'],
        ['2023', '5.8', '9.4', 'Enterprise software, analytics'],
        ['2024', '6.4', '10.3', 'Generative AI, automation']
    ]
    
    # Convert table data to Paragraphs for better text wrapping
    table1_paragraphs = []
    header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', alignment=TA_CENTER)
    for i, row in enumerate(table1_data):
        if i == 0:  # Header row
            para_row = [Paragraph(str(cell).replace('<br/>', '<br/>'), header_style) for cell in row]
        else:
            para_row = [Paragraph(str(cell), styles['Normal']) if '<br/>' in str(cell) else str(cell) for cell in row]
        table1_paragraphs.append(para_row)
    
    table1_data = table1_paragraphs
    
    table1 = Table(table1_data, colWidths=[0.8*inch, 1.6*inch, 1.0*inch, 2.1*inch])
    table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(table1)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Historical Context and Market Drivers", heading_style))
    story.append(Paragraph(
        "The technology sector's remarkable growth trajectory over the past five years reflects "
        "a fundamental shift in how businesses and consumers interact with technology. The "
        "pandemic period from 2020-2021 marked a significant acceleration in digital adoption, "
        "as organizations scrambled to implement remote work capabilities and digital business "
        "models. This period saw unprecedented growth rates, with the sector expanding by over "
        "14% in a single year.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The shift to remote work created massive demand for collaboration tools, cloud infrastructure, "
        "and cybersecurity solutions. Companies invested billions in upgrading their digital "
        "infrastructure to support distributed workforces. Video conferencing platforms, project "
        "management tools, and cloud storage services experienced explosive growth as organizations "
        "sought to maintain productivity despite physical separation.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "E-commerce saw particularly strong growth during this period, as consumers turned to "
        "online shopping out of necessity. This shift accelerated what would have otherwise "
        "taken years to achieve, compressing a decade of digital commerce evolution into a "
        "matter of months. Retail technology platforms, payment processing systems, and logistics "
        "software all benefited from this rapid transition.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "As the world emerged from the pandemic, the technology sector continued to grow, though "
        "at a more sustainable pace. The foundations laid during the crisis period have enabled "
        "ongoing digital transformation initiatives. Organizations now view technology not just "
        "as a support function, but as a core strategic enabler of business objectives.",
        normal_style
    ))
    story.append(PageBreak())
    
    # Page 2: Key Technology Segments
    story.append(Paragraph("Key Technology Segments and Market Performance", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        "The technology sector is composed of several key segments, each contributing uniquely "
        "to overall industry growth. Software and services represent the largest segment, "
        "followed by hardware, telecommunications, and emerging technologies. Understanding "
        "the dynamics of each segment provides valuable insights into the overall sector "
        "performance and future prospects.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Software and services continue to dominate the technology landscape, accounting for "
        "over one-third of total sector revenue. This segment includes enterprise software, "
        "application development, system integration, and managed services. The shift toward "
        "subscription-based software models has created predictable revenue streams for "
        "software companies while providing customers with continuous updates and support.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Enterprise resource planning (ERP) systems, customer relationship management (CRM) "
        "platforms, and human resources management software have become essential tools for "
        "modern businesses. The integration of artificial intelligence and machine learning "
        "capabilities into these platforms has added new value propositions, enabling "
        "predictive analytics, automated workflows, and personalized user experiences.",
        normal_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Table 2: Market Share by Segment
    story.append(Paragraph("Market Share by Technology Segment (2024)", heading_style))
    
    table2_data = [
        ['Segment', 'Market<br/>Share (%)', 'Revenue<br/>(Billion USD)', 'Growth<br/>Rate (%)'],
        ['Software & Services', '38', '2,432', '12.5'],
        ['Cloud Computing', '22', '1,408', '18.2'],
        ['Hardware & Devices', '18', '1,152', '6.8'],
        ['Telecommunications', '12', '768', '4.2'],
        ['Cybersecurity', '5', '320', '15.7'],
        ['AI & ML', '3', '192', '28.5'],
        ['Other', '2', '128', '8.1']
    ]
    
    # Convert table data to Paragraphs
    header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', alignment=TA_CENTER)
    table2_paragraphs = []
    for i, row in enumerate(table2_data):
        if i == 0:  # Header row
            para_row = [Paragraph(str(cell).replace('<br/>', '<br/>'), header_style) for cell in row]
        else:
            para_row = [Paragraph(str(cell), styles['Normal']) if '<br/>' in str(cell) else str(cell) for cell in row]
        table2_paragraphs.append(para_row)
    
    table2_data = table2_paragraphs
    
    table2 = Table(table2_data, colWidths=[1.6*inch, 1.2*inch, 1.5*inch, 1.2*inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(table2)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Cloud Computing Expansion", heading_style))
    story.append(Paragraph(
        "Cloud computing has emerged as one of the fastest-growing segments within the technology "
        "sector. The transition from on-premises infrastructure to cloud-based solutions has "
        "accelerated significantly over the past several years. Organizations are recognizing "
        "the benefits of cloud computing, including reduced capital expenditure, improved "
        "scalability, and enhanced flexibility.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Large cloud providers continue to invest heavily in expanding their global infrastructure, "
        "building data centers in new regions to reduce latency and comply with data residency "
        "requirements. The competition among major cloud providers has driven innovation in "
        "services, pricing models, and capabilities. Infrastructure as a Service (IaaS), Platform "
        "as a Service (PaaS), and Software as a Service (SaaS) offerings have matured significantly, "
        "making cloud adoption more accessible to organizations of all sizes.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph("Emerging Trends", heading_style))
    story.append(Paragraph(
        "Several emerging trends are shaping the future of the technology sector. Artificial "
        "intelligence and machine learning are experiencing exponential growth, particularly "
        "with the advent of generative AI technologies. Cloud computing continues to expand "
        "as organizations migrate from on-premises infrastructure to hybrid and multi-cloud "
        "environments.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Generative AI represents one of the most significant technological breakthroughs in recent "
        "years. Large language models and image generation systems have demonstrated capabilities "
        "that were previously thought to be decades away. These technologies are finding applications "
        "across numerous industries, from content creation and software development to scientific "
        "research and customer service automation.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The democratization of AI tools has enabled smaller organizations to leverage advanced "
        "machine learning capabilities without the need for large in-house data science teams. "
        "Cloud-based AI platforms provide pre-trained models and APIs that can be integrated "
        "into existing applications, reducing the barrier to entry for AI adoption.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Cybersecurity has become a top priority for organizations as the frequency and sophistication "
        "of cyber attacks continue to increase. The growing attack surface created by cloud adoption, "
        "remote work, and IoT devices has necessitated comprehensive security strategies. Organizations "
        "are investing in zero-trust architectures, advanced threat detection systems, and security "
        "awareness training to protect their digital assets.",
        normal_style
    ))
    story.append(PageBreak())
    
    # Page 3: Regional Analysis
    story.append(Paragraph("Regional Technology Market Analysis", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        "The technology sector shows varying growth patterns across different geographic regions. "
        "North America remains the dominant market, followed by Asia-Pacific, which has shown "
        "exceptional growth rates driven by innovation hubs in China, India, and Southeast Asia. "
        "Regional differences in regulatory environments, talent availability, and market maturity "
        "create unique opportunities and challenges for technology companies operating in different "
        "parts of the world.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "North America, particularly the United States, maintains its position as the global "
        "technology leader. Silicon Valley continues to be a hub of innovation and venture capital, "
        "though other technology centers have emerged throughout the region. Cities like Seattle, "
        "Austin, and Boston have developed thriving technology ecosystems with strong talent pools "
        "and supportive business environments. Canadian technology centers in Toronto, Vancouver, "
        "and Montreal have also gained prominence, attracting investment and talent from around "
        "the world.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The Asia-Pacific region's technology sector has been characterized by rapid growth and "
        "innovation. China's technology giants have become global players, competing effectively "
        "in markets worldwide. The country's large domestic market and significant investment in "
        "infrastructure have created favorable conditions for technology development. India's "
        "software services industry remains a global powerhouse, while the country is also "
        "emerging as a significant market for technology consumption.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Southeast Asian markets are experiencing particularly strong growth, driven by increasing "
        "internet penetration and smartphone adoption. Countries like Singapore, Indonesia, and "
        "Vietnam have developed vibrant startup ecosystems with active venture capital investment. "
        "These markets present opportunities for technology companies seeking to expand in "
        "high-growth emerging economies.",
        normal_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Table 3: Regional Market Analysis
    story.append(Paragraph("Regional Technology Market Overview (2024)", heading_style))
    
    table3_data = [
        ['Region', 'Market Size<br/>(Billion USD)', 'Growth<br/>Rate (%)', 'Key Markets'],
        ['North America', '2,560', '9.8', 'USA, Canada'],
        ['Asia-Pacific', '2,048', '14.2', 'China, India, Japan, SG'],
        ['Europe', '1,280', '7.5', 'UK, Germany, France'],
        ['Latin America', '384', '12.1', 'Brazil, Mexico, Argentina'],
        ['MENA', '128', '11.3', 'UAE, Israel, South Africa']
    ]
    
    # Convert table data to Paragraphs
    header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', alignment=TA_CENTER)
    table3_paragraphs = []
    for i, row in enumerate(table3_data):
        if i == 0:  # Header row
            para_row = [Paragraph(str(cell).replace('<br/>', '<br/>'), header_style) for cell in row]
        else:
            para_row = [Paragraph(str(cell), styles['Normal']) if '<br/>' in str(cell) else str(cell) for cell in row]
        table3_paragraphs.append(para_row)
    
    table3_data = table3_paragraphs
    
    table3 = Table(table3_data, colWidths=[1.3*inch, 1.5*inch, 1.2*inch, 2.0*inch])
    table3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(table3)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph(
        "The Asia-Pacific region has emerged as a major technology innovation hub, with China "
        "leading in manufacturing and hardware, while India excels in software development and "
        "IT services. The region's large population and increasing internet penetration create "
        "significant opportunities for tech companies.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Europe's technology sector, while smaller in absolute terms compared to North America "
        "and Asia-Pacific, has several notable strengths. Countries like the United Kingdom, "
        "Germany, and France have developed strong technology ecosystems with particular "
        "expertise in areas such as fintech, cybersecurity, and industrial software. European "
        "regulations, particularly the General Data Protection Regulation (GDPR), have influenced "
        "global data privacy standards and created opportunities for compliance-focused technology "
        "solutions.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Latin America's technology sector is experiencing significant growth, driven by increasing "
        "digital adoption and supportive government policies. Brazil and Mexico are the largest "
        "markets in the region, with growing startup ecosystems and increasing investment from "
        "both domestic and international sources. The region's large, young population and growing "
        "middle class present attractive opportunities for technology companies.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The Middle East and Africa region, while smaller in scale, shows promising growth "
        "potential. The United Arab Emirates and Israel have developed sophisticated technology "
        "sectors with strong innovation capabilities. Israel, in particular, has earned recognition "
        "as a global cybersecurity and software development hub. African markets are beginning to "
        "emerge as significant technology consumers, with mobile-first strategies proving effective "
        "in reaching new customer segments.",
        normal_style
    ))
    story.append(PageBreak())
    
    # Page 4: Employment and Investment
    story.append(Paragraph("Employment Trends and Investment in Technology", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        "The technology sector is one of the largest employers globally and continues to create "
        "millions of new jobs each year. However, the nature of employment is evolving, with "
        "increasing demand for specialized skills in areas such as cloud architecture, data science, "
        "and cybersecurity. The sector's job market reflects broader trends in the economy, with "
        "a shift toward knowledge work and technical expertise.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Software development remains the largest employment category within the technology sector, "
        "with millions of developers worldwide working on applications, systems, and platforms. "
        "The demand for software development skills has expanded beyond traditional technology "
        "companies, as organizations across all industries seek to build digital capabilities. "
        "Programming languages and frameworks continue to evolve, requiring developers to continuously "
        "update their skills to remain competitive in the job market.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The rise of data science and analytics has created new career paths within the technology "
        "sector. Organizations are recognizing the value of data-driven decision-making, leading to "
        "increased demand for professionals who can extract insights from large datasets. Data "
        "scientists combine statistical knowledge, programming skills, and domain expertise to "
        "solve complex business problems and identify opportunities for optimization.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Cybersecurity professionals are in particularly high demand as organizations grapple with "
        "increasingly sophisticated threats. The cybersecurity skills gap remains a significant "
        "challenge, with demand far outstripping supply. This has led to competitive compensation "
        "packages and strong career growth prospects for individuals entering the field. The "
        "breadth of cybersecurity roles ranges from technical specialists focused on specific "
        "technologies to strategic advisors who help organizations develop comprehensive security "
        "programs.",
        normal_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Table 4: Employment Statistics
    story.append(Paragraph("Technology Sector Employment Statistics", heading_style))
    
    table4_data = [
        ['Job Category', 'Employment<br/>(Millions)', 'Growth<br/>Rate (%)', 'Top Skills'],
        ['Software Dev', '26.8', '8.5', 'Programming, Agile, DevOps'],
        ['IT Support', '18.3', '4.2', 'Troubleshooting, Support'],
        ['Cybersecurity', '4.1', '12.7', 'Security, Risk Assessment'],
        ['Data Science', '3.9', '15.3', 'Python, ML, Statistics'],
        ['Cloud Architecture', '2.7', '18.9', 'AWS/Azure, Design'],
        ['AI/ML Engineering', '1.2', '24.5', 'TensorFlow, Neural Nets']
    ]
    
    # Convert table data to Paragraphs
    header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', alignment=TA_CENTER)
    table4_paragraphs = []
    for i, row in enumerate(table4_data):
        if i == 0:  # Header row
            para_row = [Paragraph(str(cell).replace('<br/>', '<br/>'), header_style) for cell in row]
        else:
            para_row = [Paragraph(str(cell), styles['Normal']) if '<br/>' in str(cell) else str(cell) for cell in row]
        table4_paragraphs.append(para_row)
    
    table4_data = table4_paragraphs
    
    table4 = Table(table4_data, colWidths=[1.5*inch, 1.3*inch, 1.2*inch, 2.0*inch])
    table4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(table4)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Remote Work and Distributed Teams", heading_style))
    story.append(Paragraph(
        "The technology sector has been at the forefront of the remote work revolution. Many "
        "technology companies have adopted flexible work policies, allowing employees to work "
        "from anywhere. This shift has expanded the talent pool available to technology companies, "
        "enabling them to hire the best candidates regardless of geographic location. However, "
        "remote work also presents challenges related to team collaboration, company culture, "
        "and employee engagement that organizations continue to navigate.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph("Venture Capital Investment", heading_style))
    story.append(Paragraph(
        "Venture capital investment in technology startups has remained robust, with particular "
        "focus on artificial intelligence, fintech, and healthcare technology. The availability "
        "of capital continues to fuel innovation and startup formation across various tech "
        "subsectors. Despite periodic market corrections, the long-term trend in venture capital "
        "investment remains positive.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Early-stage technology startups continue to attract significant funding, particularly "
        "in areas such as artificial intelligence, blockchain, and clean technology. Investors "
        "are seeking companies with strong technical differentiation, large addressable markets, "
        "and capable founding teams. The competitive landscape for funding has intensified, with "
        "founders needing to demonstrate clear value propositions and viable paths to profitability.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Corporate venture capital has become an increasingly important source of funding for "
        "technology startups. Large technology companies and traditional enterprises are investing "
        "in startups to gain access to innovative technologies and business models. These strategic "
        "investments often include commercial relationships that can accelerate startup growth while "
        "providing corporations with cutting-edge capabilities.",
        normal_style
    ))
    story.append(PageBreak())
    
    # Page 5: Future Outlook
    story.append(Paragraph("Future Outlook and Predictions", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        "Looking ahead, the technology sector is poised for continued growth, driven by several "
        "key factors. The proliferation of Internet of Things (IoT) devices, the expansion of "
        "5G networks, and the maturation of artificial intelligence technologies are expected "
        "to create new opportunities and markets. These trends are converging to enable entirely "
        "new classes of applications and services.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The Internet of Things is creating unprecedented connectivity between devices, systems, "
        "and environments. Smart cities, connected vehicles, and industrial IoT applications are "
        "generating vast amounts of data that can be analyzed to improve efficiency, safety, and "
        "user experiences. As IoT deployments scale, the need for edge computing capabilities "
        "becomes more critical, enabling real-time processing and decision-making closer to where "
        "data is generated.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Fifth-generation wireless networks (5G) are enabling new use cases that were previously "
        "not feasible with older network technologies. The low latency and high bandwidth of 5G "
        "networks support applications such as autonomous vehicles, remote surgery, and augmented "
        "reality experiences. As 5G infrastructure continues to expand globally, new business "
        "models and services are emerging that leverage these enhanced network capabilities.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Artificial intelligence is evolving from a specialized technology to a fundamental capability "
        "embedded across software and services. Machine learning models are becoming more efficient "
        "and accessible, enabling organizations of all sizes to leverage AI capabilities. The "
        "combination of improved algorithms, increased computational power, and better data "
        "availability is driving rapid advancement in AI capabilities across multiple domains.",
        normal_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Table 5: Projected Growth by Technology Area
    story.append(Paragraph("Projected Growth by Technology Area (2025-2027)", heading_style))
    
    table5_data = [
        ['Technology Area', '2025 Forecast<br/>(Billion USD)', '2027 Forecast<br/>(Billion USD)', 'CAGR (%)'],
        ['Artificial Intelligence', '450', '680', '23.0'],
        ['Cloud Computing', '620', '890', '19.8'],
        ['Cybersecurity', '180', '260', '20.2'],
        ['Edge Computing', '95', '165', '31.7'],
        ['Blockchain', '45', '78', '31.5'],
        ['Quantum Computing', '8', '18', '50.0'],
        ['IoT Solutions', '220', '340', '24.4']
    ]
    
    # Convert table data to Paragraphs
    header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', alignment=TA_CENTER)
    table5_paragraphs = []
    for i, row in enumerate(table5_data):
        if i == 0:  # Header row
            para_row = [Paragraph(str(cell).replace('<br/>', '<br/>'), header_style) for cell in row]
        else:
            para_row = [Paragraph(str(cell), styles['Normal']) if '<br/>' in str(cell) else str(cell) for cell in row]
        table5_paragraphs.append(para_row)
    
    table5_data = table5_paragraphs
    
    table5 = Table(table5_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.0*inch])
    table5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(table5)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Key Challenges and Opportunities", heading_style))
    story.append(Paragraph(
        "While the technology sector presents tremendous opportunities, it also faces significant "
        "challenges. Cybersecurity threats continue to evolve, requiring constant vigilance and "
        "investment in protective measures. Regulatory compliance across different jurisdictions "
        "presents complexity for global tech companies. Additionally, the need for skilled talent "
        "remains a persistent challenge that affects organizations across all segments of the "
        "technology sector.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The cybersecurity landscape is becoming increasingly complex as attack surfaces expand "
        "and threat actors become more sophisticated. Ransomware attacks, data breaches, and "
        "nation-state cyber operations pose significant risks to organizations. The cost of "
        "cybersecurity incidents continues to rise, driving increased investment in protective "
        "technologies and processes. However, there is often a gap between security capabilities "
        "and the threats organizations face, creating ongoing challenges for cybersecurity "
        "professionals.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Regulatory environments are becoming more complex as governments seek to address concerns "
        "about data privacy, platform competition, and content moderation. Technology companies "
        "must navigate varying requirements across different jurisdictions, increasing compliance "
        "costs and complexity. The European Union's Digital Services Act and Digital Markets Act "
        "represent significant regulatory changes that will affect how technology platforms operate. "
        "Similar regulatory initiatives are emerging in other regions, creating a fragmented "
        "regulatory landscape.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "However, these challenges also create opportunities. The demand for cybersecurity solutions "
        "drives innovation in that space, with new technologies and services emerging to address "
        "evolving threats. Regulatory technology (RegTech) has emerged as a growing sector, "
        "providing software and services to help organizations manage compliance requirements "
        "more efficiently. Educational technology and online learning platforms are addressing "
        "the skills gap while creating new business models for training and professional development.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "Sustainability and environmental considerations are becoming increasingly important in "
        "the technology sector. Data centers consume significant amounts of energy, and the "
        "manufacturing of electronic devices has environmental impacts. Technology companies "
        "are investing in renewable energy, improving data center efficiency, and developing "
        "more sustainable manufacturing processes. These initiatives create opportunities for "
        "clean technology innovation while addressing environmental concerns.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "The convergence of technologies is creating new possibilities for innovation. The "
        "combination of artificial intelligence, edge computing, and 5G networks enables "
        "applications that were previously impossible. Autonomous systems, smart infrastructure, "
        "and immersive digital experiences represent just a few of the areas where technological "
        "convergence is driving innovation.",
        normal_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        "In conclusion, the technology sector remains at the forefront of economic growth and "
        "innovation. With continued investment in research and development, emphasis on talent "
        "development, and adaptive business strategies, the sector is well-positioned to maintain "
        "its trajectory of expansion and transformation in the years to come. The challenges "
        "facing the sector, while significant, are being addressed through innovation and "
        "collaboration, ensuring that technology continues to drive progress across all aspects "
        "of human endeavor.",
        normal_style
    ))
    
    # Build PDF
    doc.build(story)
    print(f"PDF successfully generated: {output_path}")


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "tech_sector_report.pdf"
    create_tech_sector_pdf(output_path)

