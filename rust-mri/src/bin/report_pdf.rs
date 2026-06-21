use printpdf::*;
use std::fs::File;
use std::io::{self, Read, BufWriter};

#[derive(Debug, Default)]
struct RadiologyReport {
    patient_name: String,
    patient_age: String,
    patient_sex: String,
    date: String,
    referring_physician: String,
    report_id: String,
    clinical_indication: String,
    technique: String,
    findings: String,
    impression: Vec<String>,
    recommendation: String,
}

fn mm(val: f32) -> Mm {
    Mm(val / 2.83464)
}

fn pt(x: f32, y: f32) -> (Point, bool) {
    (Point::new(mm(x), mm(y)), false)
}

fn wrap_text(text: &str, max_width_points: f32, font_size: f32, is_bold: bool) -> Vec<String> {
    let avg_char_width = if is_bold { 0.45 } else { 0.40 } * font_size;
    let max_chars_per_line = (max_width_points / avg_char_width).floor() as usize;

    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let p_trimmed = paragraph.trim();
        if p_trimmed.is_empty() {
            continue;
        }
        let mut current_line = String::new();
        for word in p_trimmed.split_whitespace() {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else if current_line.len() + 1 + word.len() <= max_chars_per_line {
                current_line.push(' ');
                current_line.push_str(word);
            } else {
                lines.push(current_line);
                current_line = word.to_string();
            }
        }
        if !current_line.is_empty() {
            lines.push(current_line);
        }
    }
    lines
}

fn parse_report(content: &str) -> RadiologyReport {
    let mut report = RadiologyReport::default();
    
    // Default values
    report.patient_name = "Unknown".to_string();
    report.patient_age = "Unknown".to_string();
    report.patient_sex = "Unknown".to_string();
    report.date = "Unknown".to_string();
    report.referring_physician = "Dr. A. Sharma".to_string();
    report.report_id = "RAD-00142".to_string();

    let mut current_section = "";
    let mut clinical_lines = Vec::new();
    let mut technique_lines = Vec::new();
    let mut findings_lines = Vec::new();
    let mut impression_lines = Vec::new();
    let mut recommendation_lines = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse Patient details
        if trimmed.starts_with("Patient:") {
            let parts: Vec<&str> = trimmed.split('|').collect();
            for part in parts {
                let kv: Vec<&str> = part.split(':').collect();
                if kv.len() >= 2 {
                    let k = kv[0].trim().to_lowercase();
                    let v = kv[1..].join(":").trim().to_string();
                    if k.contains("patient") {
                        report.patient_name = v;
                    } else if k.contains("age") {
                        report.patient_age = v;
                    } else if k.contains("sex") {
                        report.patient_sex = v;
                    }
                }
            }
            continue;
        }

        if trimmed.starts_with("Date:") {
            if let Some(val) = trimmed.strip_prefix("Date:") {
                report.date = val.trim().to_string();
            }
            continue;
        }
        
        if trimmed.starts_with("Report ID:") {
            if let Some(val) = trimmed.strip_prefix("Report ID:") {
                report.report_id = val.trim().to_string();
            }
            continue;
        }

        let upper = trimmed.to_uppercase();
        if upper.contains("CLINICAL INDICATION") {
            current_section = "clinical";
            continue;
        } else if upper.contains("TECHNIQUE") {
            current_section = "technique";
            continue;
        } else if upper.contains("FINDINGS") {
            current_section = "findings";
            continue;
        } else if upper.contains("IMPRESSION") {
            current_section = "impression";
            continue;
        } else if upper.contains("RECOMMENDATION") {
            current_section = "recommendation";
            continue;
        } else if upper.starts_with("---") || upper.starts_with("RADIOLOGY REPORT") {
            continue;
        }

        match current_section {
            "clinical" => clinical_lines.push(trimmed),
            "technique" => technique_lines.push(trimmed),
            "findings" => findings_lines.push(trimmed),
            "impression" => impression_lines.push(trimmed),
            "recommendation" => recommendation_lines.push(trimmed),
            _ => {}
        }
    }

    report.clinical_indication = clinical_lines.join(" ");
    report.technique = technique_lines.join(" ");
    report.findings = findings_lines.join(" ");
    report.recommendation = recommendation_lines.join(" ");

    // Clean impression numbered list lines
    for line in impression_lines {
        let mut clean = line;
        if let Some(idx) = clean.find('.') {
            if idx < 3 {
                clean = &clean[idx + 1..];
            }
        }
        let clean_trimmed = clean.trim();
        let final_line = if clean_trimmed.starts_with('*') || clean_trimmed.starts_with('-') {
            clean_trimmed[1..].trim().to_string()
        } else {
            clean_trimmed.to_string()
        };
        if !final_line.is_empty() {
            report.impression.push(final_line);
        }
    }

    report
}

fn draw_rect_filled(layer: &PdfLayerReference, x: f32, y: f32, w: f32, h: f32, r: f32, g: f32, b: f32) {
    layer.set_fill_color(Color::Rgb(Rgb::new(r, g, b, None)));
    layer.set_outline_color(Color::Rgb(Rgb::new(r, g, b, None)));
    let points = vec![
        pt(x, y),
        pt(x + w, y),
        pt(x + w, y + h),
        pt(x, y + h),
    ];
    let poly = Polygon {
        rings: vec![points],
        mode: printpdf::path::PaintMode::Fill,
        winding_order: printpdf::path::WindingOrder::NonZero,
    };
    layer.add_polygon(poly);
}

fn draw_line(layer: &PdfLayerReference, x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, r: f32, g: f32, b: f32) {
    layer.set_outline_color(Color::Rgb(Rgb::new(r, g, b, None)));
    layer.set_outline_thickness(thickness);
    let line = Line {
        points: vec![pt(x1, y1), pt(x2, y2)],
        is_closed: false,
    };
    layer.add_line(line);
}

fn draw_text_line(layer: &PdfLayerReference, text: &str, font: &IndirectFontRef, size: f32, x: f32, y: f32, r: f32, g: f32, b: f32) {
    layer.begin_text_section();
    layer.set_fill_color(Color::Rgb(Rgb::new(r, g, b, None)));
    layer.set_font(font, size);
    layer.set_text_cursor(mm(x), mm(y));
    layer.write_text(text, font);
    layer.end_text_section();
}

fn draw_section_header(
    layer: &PdfLayerReference,
    font_bold: &IndirectFontRef,
    label: &str,
    y: f32,
) {
    // Mid blue text: #185FA5 (24, 95, 165)
    draw_text_line(layer, &label.to_uppercase(), font_bold, 10.0, 40.0, y, 24.0/255.0, 95.0/255.0, 165.0/255.0);
    // Underline
    draw_line(layer, 40.0, y - 4.0, 555.0, y - 4.0, 1.5, 24.0/255.0, 95.0/255.0, 165.0/255.0);
}

fn draw_text_block(
    layer: &PdfLayerReference,
    font: &IndirectFontRef,
    text: &str,
    start_y: f32,
    line_height: f32,
    font_size: f32,
    max_width: f32,
) -> f32 {
    let lines = wrap_text(text, max_width, font_size, false);
    
    let mut current_y = start_y;
    for line in lines {
        // Black: #042C53 (4, 44, 83)
        draw_text_line(layer, &line, font, font_size, 40.0, current_y, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        current_y -= line_height;
    }
    current_y
}

fn draw_standard_section(
    layer: &PdfLayerReference,
    font_regular: &IndirectFontRef,
    font_bold: &IndirectFontRef,
    label: &str,
    text: &str,
    start_y: f32,
    max_width: f32,
) -> f32 {
    draw_section_header(layer, font_bold, label, start_y);
    let text_start_y = start_y - 20.0;
    let end_y = draw_text_block(layer, font_regular, text, text_start_y, 16.0, 13.0, max_width);
    end_y
}

fn draw_imaging_section(
    layer: &PdfLayerReference,
    font_regular: &IndirectFontRef,
    font_bold: &IndirectFontRef,
    start_y: f32,
    _max_width: f32,
) -> f32 {
    draw_section_header(layer, font_bold, "Imaging", start_y);
    let box_top = start_y - 12.0;
    let box_height = 140.0;
    let box_bottom = box_top - box_height;
    
    // Draw background (Light blue: #E6F1FB)
    draw_rect_filled(layer, 40.0, box_bottom, 515.0, box_height, 230.0/255.0, 241.0/255.0, 251.0/255.0);

    // Draw dashed border (#85B7EB -> 133, 183, 235)
    layer.set_outline_color(Color::Rgb(Rgb::new(133.0 / 255.0, 183.0 / 255.0, 235.0 / 255.0, None)));
    layer.set_outline_thickness(1.5);
    
    let dash_pattern = LineDashPattern {
        offset: 0,
        dash_1: Some(4),
        gap_1: Some(4),
        dash_2: None,
        gap_2: None,
        dash_3: None,
        gap_3: None,
    };
    layer.set_line_dash_pattern(dash_pattern);
    
    let border_points = vec![
        pt(40.0, box_bottom),
        pt(555.0, box_bottom),
        pt(555.0, box_top),
        pt(40.0, box_top),
    ];
    layer.add_polygon(Polygon {
        rings: vec![border_points],
        mode: printpdf::path::PaintMode::Stroke,
        winding_order: printpdf::path::WindingOrder::NonZero,
    });
    
    // Restore solid line
    layer.set_line_dash_pattern(LineDashPattern {
        offset: 0,
        dash_1: None,
        gap_1: None,
        dash_2: None,
        gap_2: None,
        dash_3: None,
        gap_3: None,
    });

    // Write text placeholders centered inside the box
    // Icon placeholder
    draw_text_line(layer, "[o]", font_bold, 20.0, 280.0, box_top - 50.0, 55.0/255.0, 138.0/255.0, 221.0/255.0);
    // "MRI scan image"
    draw_text_line(layer, "MRI scan image", font_bold, 12.0, 245.0, box_top - 80.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
    // "Image will be inserted here"
    draw_text_line(layer, "Image will be inserted here", font_regular, 11.0, 215.0, box_top - 105.0, 55.0/255.0, 138.0/255.0, 221.0/255.0);

    box_bottom
}

fn draw_impression_section(
    layer: &PdfLayerReference,
    font_regular: &IndirectFontRef,
    font_bold: &IndirectFontRef,
    items: &[String],
    start_y: f32,
    max_width: f32,
) -> f32 {
    draw_section_header(layer, font_bold, "Impression", start_y);
    let box_top = start_y - 12.0;

    let line_height = 16.0;
    let text_margin = 12.0;
    let list_x = 60.0;
    let list_max_width = max_width - 30.0;
    
    let mut wrapped_items = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let full_text = format!("{}. {}", i + 1, item);
        let lines = wrap_text(&full_text, list_max_width, 13.0, false);
        wrapped_items.push(lines);
    }
    
    let total_lines: usize = wrapped_items.iter().map(|item| item.len()).sum();
    let spacing_between_items = 4.0;
    let box_height = total_lines as f32 * line_height 
        + (items.len().saturating_sub(1)) as f32 * spacing_between_items
        + 2.0 * text_margin;
        
    let box_bottom = box_top - box_height;

    // Draw background (Light blue)
    draw_rect_filled(layer, 40.0, box_bottom, 515.0, box_height, 230.0/255.0, 241.0/255.0, 251.0/255.0);

    // Draw left Mid-blue border accent (3px solid #185FA5 -> 24, 95, 165)
    draw_rect_filled(layer, 40.0, box_bottom, 3.0, box_height, 24.0/255.0, 95.0/255.0, 165.0/255.0);

    // Draw list items
    let mut current_y = box_top - text_margin - 11.0;
    for lines in wrapped_items {
        for line in lines {
            draw_text_line(layer, &line, font_regular, 13.0, list_x, current_y, 4.0/255.0, 44.0/255.0, 83.0/255.0);
            current_y -= line_height;
        }
        current_y -= spacing_between_items;
    }

    box_bottom
}

fn draw_footer(
    layer: &PdfLayerReference,
    font_regular: &IndirectFontRef,
) {
    // 0.5px blue separator line at y = 75.0
    draw_line(layer, 40.0, 75.0, 555.0, 75.0, 0.5, 181.0/255.0, 212.0/255.0, 244.0/255.0);

    // Left clinic review text
    draw_text_line(layer, "Generated by Radiology Agent · For clinical review only", font_regular, 10.0, 40.0, 58.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
    
    // Right signature line
    draw_line(layer, 395.0, 58.0, 555.0, 58.0, 0.5, 55.0/255.0, 138.0/255.0, 221.0/255.0);

    // Right signature label
    draw_text_line(layer, "Reporting radiologist signature", font_regular, 8.5, 395.0, 46.0, 55.0/255.0, 138.0/255.0, 221.0/255.0);
}

struct DocState {
    doc: PdfDocumentReference,
    pages: Vec<(PdfPageIndex, PdfLayerIndex)>,
    current_page_idx: usize,
    font_regular: IndirectFontRef,
    font_bold: IndirectFontRef,
    y_cursor: f32,
}

impl DocState {
    fn current_layer(&self) -> PdfLayerReference {
        let (page_idx, layer_idx) = self.pages[self.current_page_idx];
        self.doc.get_page(page_idx).get_layer(layer_idx)
    }
    
    fn ensure_space(&mut self, needed: f32, report: &RadiologyReport) {
        if self.y_cursor - needed < 90.0 {
            // Draw footer on current page
            draw_footer(&self.current_layer(), &self.font_regular);
            
            // Add a new page
            let (page, layer) = self.doc.add_page(Mm(210.0), Mm(297.0), "Layer 1");
            self.pages.push((page, layer));
            self.current_page_idx += 1;
            self.y_cursor = 802.0;
            
            // Draw header bar on new page
            self.draw_header_bar(report);
            self.y_cursor -= 15.0;
        }
    }
    
    fn draw_header_bar(&mut self, _report: &RadiologyReport) {
        let layer = self.current_layer();
        // Background
        draw_rect_filled(&layer, 40.0, self.y_cursor - 40.0, 515.0, 40.0, 12.0/255.0, 68.0/255.0, 124.0/255.0);

        // Left text
        draw_text_line(&layer, "Radiology Report (Continued)", &self.font_bold, 11.0, 55.0, self.y_cursor - 25.0, 1.0, 1.0, 1.0);
        
        self.y_cursor -= 40.0;
    }
}

fn estimate_standard_section_height(text: &str, max_width: f32, font_size: f32, line_height: f32) -> f32 {
    let lines = wrap_text(text, max_width, font_size, false);
    lines.len() as f32 * line_height + 25.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    let mut input_path = String::new();
    let mut output_path = String::new();
    let mut name_override: Option<String> = None;
    let mut age_override: Option<String> = None;
    let mut sex_override: Option<String> = None;
    let mut physician_override: Option<String> = None;
    let mut id_override: Option<String> = None;
    let mut date_override: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                if i + 1 < args.len() {
                    input_path = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --input requires a path");
                    std::process::exit(1);
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --output requires a path");
                    std::process::exit(1);
                }
            }
            "--name" => {
                if i + 1 < args.len() {
                    name_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--age" => {
                if i + 1 < args.len() {
                    age_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--sex" => {
                if i + 1 < args.len() {
                    sex_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--physician" => {
                if i + 1 < args.len() {
                    physician_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--id" => {
                if i + 1 < args.len() {
                    id_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--date" => {
                if i + 1 < args.len() {
                    date_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            _ => {
                if input_path.is_empty() {
                    input_path = args[i].clone();
                } else if output_path.is_empty() {
                    output_path = args[i].clone();
                }
                i += 1;
            }
        }
    }

    if input_path.is_empty() || output_path.is_empty() {
        eprintln!("Usage: report_pdf --input <input_report.txt> --output <output.pdf> [overrides]");
        std::process::exit(1);
    }

    // Read report text
    let mut content = String::new();
    if input_path == "-" {
        io::stdin().read_to_string(&mut content)?;
    } else {
        let mut file = File::open(&input_path)?;
        file.read_to_string(&mut content)?;
    }

    let mut report = parse_report(&content);

    // Apply overrides if provided
    if let Some(n) = name_override { report.patient_name = n; }
    if let Some(a) = age_override { report.patient_age = a; }
    if let Some(s) = sex_override { report.patient_sex = s; }
    if let Some(p) = physician_override { report.referring_physician = p; }
    if let Some(id) = id_override { report.report_id = id; }
    if let Some(d) = date_override { report.date = d; }

    // Initialize printpdf Document
    // Mm(210.0) x Mm(297.0) is standard A4 size
    let (doc, page, layer) = PdfDocument::new("Radiology Report", Mm(210.0), Mm(297.0), "Layer 1");
    
    let font_regular = if let Ok(file) = File::open("/usr/share/fonts/TTF/DejaVuSans.ttf") {
        doc.add_external_font(file).unwrap_or_else(|_| doc.add_builtin_font(BuiltinFont::Helvetica).unwrap())
    } else {
        doc.add_builtin_font(BuiltinFont::Helvetica)?
    };

    let font_bold = if let Ok(file) = File::open("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf") {
        doc.add_external_font(file).unwrap_or_else(|_| doc.add_builtin_font(BuiltinFont::HelveticaBold).unwrap())
    } else {
        doc.add_builtin_font(BuiltinFont::HelveticaBold)?
    };

    let mut state = DocState {
        doc,
        pages: vec![(page, layer)],
        current_page_idx: 0,
        font_regular,
        font_bold,
        y_cursor: 802.0,
    };

    // 1. Draw HEADER BAR on Page 1
    {
        let layer = state.current_layer();
        // Background: Primary blue #0C447C (12, 68, 124)
        draw_rect_filled(&layer, 40.0, 742.0, 515.0, 60.0, 12.0/255.0, 68.0/255.0, 124.0/255.0);

        // Header Text
        draw_text_line(&layer, "Radiology Report", &state.font_bold, 14.0, 55.0, 775.0, 1.0, 1.0, 1.0);
        draw_text_line(&layer, "AI-assisted diagnostic report", &state.font_regular, 9.0, 55.0, 758.0, 1.0, 1.0, 1.0);
        draw_text_line(&layer, &format!("Date: {}", report.date), &state.font_regular, 10.0, 440.0, 775.0, 1.0, 1.0, 1.0);
        draw_text_line(&layer, &format!("Report ID: {}", report.report_id), &state.font_regular, 10.0, 440.0, 758.0, 1.0, 1.0, 1.0);
    }
    
    // 2. Draw PATIENT BAR on Page 1
    {
        let layer = state.current_layer();
        // Light blue background (#E6F1FB)
        draw_rect_filled(&layer, 40.0, 697.0, 515.0, 35.0, 230.0/255.0, 241.0/255.0, 251.0/255.0);

        // Accent border (3px solid #185FA5)
        draw_rect_filled(&layer, 40.0, 697.0, 3.0, 35.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);

        // Col 1: Patient name
        draw_text_line(&layer, "Patient name", &state.font_regular, 8.5, 55.0, 719.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
        draw_text_line(&layer, &report.patient_name, &state.font_bold, 10.5, 55.0, 705.0, 4.0/255.0, 44.0/255.0, 83.0/255.0);

        // Col 2: Age / Sex
        draw_text_line(&layer, "Age / Sex", &state.font_regular, 8.5, 220.0, 719.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
        let age_sex = format!("{} / {}", report.patient_age, report.patient_sex);
        draw_text_line(&layer, &age_sex, &state.font_bold, 10.5, 220.0, 705.0, 4.0/255.0, 44.0/255.0, 83.0/255.0);

        // Col 3: Referring physician
        draw_text_line(&layer, "Referring physician", &state.font_regular, 8.5, 385.0, 719.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
        draw_text_line(&layer, &report.referring_physician, &state.font_bold, 10.5, 385.0, 705.0, 4.0/255.0, 44.0/255.0, 83.0/255.0);
    }
    
    state.y_cursor = 675.0;

    // 3. Draw BODY SECTIONS dynamically
    
    // a. Clinical indication
    let height_a = estimate_standard_section_height(&report.clinical_indication, 515.0, 13.0, 16.0);
    state.ensure_space(height_a, &report);
    state.y_cursor = draw_standard_section(
        &state.current_layer(),
        &state.font_regular,
        &state.font_bold,
        "Clinical Indication",
        &report.clinical_indication,
        state.y_cursor,
        515.0,
    ) - 15.0;

    // b. Technique
    let height_b = estimate_standard_section_height(&report.technique, 515.0, 13.0, 16.0);
    state.ensure_space(height_b, &report);
    state.y_cursor = draw_standard_section(
        &state.current_layer(),
        &state.font_regular,
        &state.font_bold,
        "Technique",
        &report.technique,
        state.y_cursor,
        515.0,
    ) - 15.0;

    // c. Imaging
    let height_c = 140.0 + 25.0;
    state.ensure_space(height_c, &report);
    state.y_cursor = draw_imaging_section(
        &state.current_layer(),
        &state.font_regular,
        &state.font_bold,
        state.y_cursor,
        515.0,
    ) - 15.0;

    // d. Findings
    let height_d = estimate_standard_section_height(&report.findings, 515.0, 13.0, 16.0);
    state.ensure_space(height_d, &report);
    state.y_cursor = draw_standard_section(
        &state.current_layer(),
        &state.font_regular,
        &state.font_bold,
        "Findings",
        &report.findings,
        state.y_cursor,
        515.0,
    ) - 15.0;

    // e. Impression
    let mut total_lines = 0;
    for (i, item) in report.impression.iter().enumerate() {
        let full_text = format!("{}. {}", i + 1, item);
        let lines = wrap_text(&full_text, 485.0, 13.0, false);
        total_lines += lines.len();
    }
    let impression_box_height = total_lines as f32 * 16.0 
        + (report.impression.len().saturating_sub(1)) as f32 * 4.0
        + 24.0;
    let height_e = impression_box_height + 25.0;
    state.ensure_space(height_e, &report);
    state.y_cursor = draw_impression_section(
        &state.current_layer(),
        &state.font_regular,
        &state.font_bold,
        &report.impression,
        state.y_cursor,
        515.0,
    ) - 15.0;

    // f. Recommendation
    let height_f = estimate_standard_section_height(&report.recommendation, 515.0, 13.0, 16.0);
    state.ensure_space(height_f, &report);
    state.y_cursor = draw_standard_section(
        &state.current_layer(),
        &state.font_regular,
        &state.font_bold,
        "Recommendation",
        &report.recommendation,
        state.y_cursor,
        515.0,
    ) - 15.0;

    // 4. Draw FOOTER on the last page
    draw_footer(&state.current_layer(), &state.font_regular);

    // Save document
    let file = File::create(&output_path)?;
    state.doc.save(&mut BufWriter::new(file))?;
    
    println!("Success: PDF report generated at {}", output_path);
    Ok(())
}
