extern crate image as img_crate;
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
    patient_id: String,
    study_date: String,
    modality: String,
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
    // Use a slightly conservative char-width estimate so text stays within bounds.
    let avg_char_width = if is_bold { 0.52 } else { 0.48 } * font_size;
    let max_chars_per_line = ((max_width_points / avg_char_width).floor() as usize).max(1);

    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let p_trimmed = paragraph.trim();
        if p_trimmed.is_empty() {
            continue;
        }
        let mut current_line = String::new();
        for word in p_trimmed.split_whitespace() {
            // Hard-break words that are longer than the entire line width.
            let mut remaining = word;
            while remaining.len() > max_chars_per_line {
                let split_at = max_chars_per_line.saturating_sub(1);
                if !current_line.is_empty() {
                    lines.push(current_line.clone());
                    current_line.clear();
                }
                current_line.push_str(&remaining[..split_at]);
                current_line.push('-');
                lines.push(current_line.clone());
                current_line.clear();
                remaining = &remaining[split_at..];
            }
            let chunk = remaining;
            if current_line.is_empty() {
                current_line = chunk.to_string();
            } else if current_line.len() + 1 + chunk.len() <= max_chars_per_line {
                current_line.push(' ');
                current_line.push_str(chunk);
            } else {
                lines.push(current_line.clone());
                current_line = chunk.to_string();
            }
        }
        if !current_line.is_empty() {
            lines.push(current_line);
        }
    }
    lines
}

/// Truncate a string to at most `max_chars` characters, appending "…" if trimmed.
fn truncate_text(text: &str, max_width_points: f32, font_size: f32) -> String {
    let avg_char_width = 0.48 * font_size;
    let max_chars = ((max_width_points / avg_char_width).floor() as usize).max(1);
    if text.len() <= max_chars {
        text.to_string()
    } else {
        // Back off by 1 to leave room for the ellipsis
        let trimmed = &text[..max_chars.saturating_sub(1)];
        format!("{}…", trimmed.trim_end())
    }
}

fn clean_markdown(s: &str) -> String {
    s.replace("**", "").replace("*", "")
}

fn parse_report(content: &str) -> RadiologyReport {
    let mut report = RadiologyReport::default();
    
    // Default values
    report.patient_name = "Unknown".to_string();
    report.patient_age = "Unknown".to_string();
    report.patient_sex = "Unknown".to_string();
    report.date = "Unknown".to_string();
    report.referring_physician = "Dr. Tarun Ahuja, MD".to_string();
    report.report_id = "RAD-00142".to_string();
    report.patient_id = "Unknown".to_string();
    report.study_date = "Unknown".to_string();
    report.modality = "MRI (3T)".to_string();

    let mut current_section = "clinical";
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

        let clean_upper = trimmed.replace('*', "").replace('#', "").replace(':', "").trim().to_uppercase();
        if clean_upper == "CLINICAL INDICATION" || clean_upper == "PATIENT-FRIENDLY MRI SUMMARY" {
            current_section = "clinical";
            continue;
        } else if clean_upper == "TECHNIQUE" || clean_upper == "EXPLANATION OF TECH" {
            current_section = "technique";
            continue;
        } else if clean_upper == "FINDINGS" || clean_upper == "DETAILED EXPLANATION" {
            current_section = "findings";
            continue;
        } else if clean_upper == "IMPRESSION" || clean_upper == "WHAT WAS FOUND" {
            current_section = "impression";
            continue;
        } else if clean_upper == "RECOMMENDATION" || clean_upper == "NEXT STEPS & RECOMMENDATIONS" || clean_upper == "NEXT STEPS AND RECOMMENDATIONS" {
            current_section = "recommendation";
            continue;
        } else if clean_upper.starts_with("---") || clean_upper.starts_with("RADIOLOGY REPORT") {
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

    report.clinical_indication = clinical_lines.join("\n");
    report.technique = technique_lines.join("\n");
    report.findings = findings_lines.join("\n");
    report.recommendation = recommendation_lines.join("\n");

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
        let final_cleaned = final_line;
        if !final_cleaned.is_empty() {
            report.impression.push(final_cleaned);
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

fn draw_rich_text_line(
    layer: &PdfLayerReference,
    text: &str,
    font_regular: &IndirectFontRef,
    font_bold: &IndirectFontRef,
    size: f32,
    x: f32,
    y: f32,
    r: f32,
    g: f32,
    b: f32,
) {
    layer.begin_text_section();
    layer.set_fill_color(Color::Rgb(Rgb::new(r, g, b, None)));
    layer.set_text_cursor(mm(x), mm(y));
    
    let parts = text.split("**");
    for (idx, part) in parts.enumerate() {
        let is_bold = idx % 2 == 1;
        let font = if is_bold { font_bold } else { font_regular };
        layer.set_font(font, size);
        let cleaned_part = part.replace("*", "");
        layer.write_text(&cleaned_part, font);
    }
    layer.end_text_section();
}

fn draw_formatted_text_block(
    state: &mut DocState,
    report: &RadiologyReport,
    text: &str,
    max_width: f32,
    font_size: f32,
    line_height: f32,
) {
    for paragraph in text.split('\n') {
        let p_trimmed = paragraph.trim();
        if p_trimmed.is_empty() {
            state.y_cursor -= 4.0;
            continue;
        }

        let mut prefix = "";
        let mut rest = p_trimmed;
        let mut is_list = false;

        if p_trimmed.starts_with("- ") {
            prefix = "• ";
            rest = &p_trimmed[2..];
            is_list = true;
        } else if p_trimmed.starts_with("* ") {
            prefix = "• ";
            rest = &p_trimmed[2..];
            is_list = true;
        } else if p_trimmed.starts_with("• ") {
            prefix = "• ";
            rest = &p_trimmed[2..];
            is_list = true;
        } else if let Some(idx) = p_trimmed.find('.') {
            if idx > 0 && idx < 3 {
                let num_str = &p_trimmed[..idx];
                if num_str.chars().all(|c| c.is_ascii_digit()) {
                    prefix = &p_trimmed[..idx + 2];
                    rest = &p_trimmed[idx + 1..].trim();
                    is_list = true;
                }
            }
        }

        // List items: bullet at x=42, text at x=55 — shrink width by 15 to match indent
        let (x_start, width_limit) = if is_list {
            (55.0, max_width - 15.0)
        } else {
            (40.0, max_width)
        };

        let lines = wrap_text(rest, width_limit, font_size, false);

        for (line_idx, line) in lines.iter().enumerate() {
            state.ensure_space(line_height + 4.0, report);

            let layer = state.current_layer();

            if line_idx == 0 && is_list {
                draw_rich_text_line(
                    &layer,
                    prefix,
                    &state.font_regular,
                    &state.font_bold,
                    font_size,
                    42.0,
                    state.y_cursor,
                    24.0/255.0, 95.0/255.0, 165.0/255.0
                );
            }

            draw_rich_text_line(
                &layer,
                line,
                &state.font_regular,
                &state.font_bold,
                font_size,
                x_start,
                state.y_cursor,
                4.0/255.0, 44.0/255.0, 83.0/255.0
            );

            state.y_cursor -= line_height;
        }
    }
}

fn draw_standard_section(
    state: &mut DocState,
    report: &RadiologyReport,
    label: &str,
    text: &str,
    max_width: f32,
) {
    state.ensure_space(25.0, report);
    draw_section_header(&state.current_layer(), &state.font_bold, label, state.y_cursor);
    state.y_cursor -= 20.0;
    draw_formatted_text_block(state, report, text, max_width, 13.0, 16.0);
}

fn draw_image_at(
    layer: &PdfLayerReference,
    image_path: &str,
    x: f32,
    y: f32,
    width_pt: f32,
    height_pt: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::BufReader;
    use img_crate::codecs::png::PngDecoder;

    let img = img_crate::open(image_path)?;
    let w_px = img.width() as f32;
    let h_px = img.height() as f32;

    let file = File::open(image_path)?;
    let reader = BufReader::new(file);
    let decoder = PngDecoder::new(reader)?;
    let xobject = ImageXObject::try_from(decoder)?;
    let image = Image::from(xobject);

    image.add_to_layer(
        layer.clone(),
        ImageTransform {
            translate_x: Some(Mm(x * 0.352778)),
            translate_y: Some(Mm((y - height_pt) * 0.352778)),
            rotate: None,
            scale_x: Some(width_pt / w_px),
            scale_y: Some(height_pt / h_px),
            dpi: Some(72.0),
        },
    );
    Ok(())
}

fn draw_imaging_section(
    layer: &PdfLayerReference,
    font_regular: &IndirectFontRef,
    font_bold: &IndirectFontRef,
    mri_path: Option<&str>,
    kspace_path: Option<&str>,
    start_y: f32,
    _max_width: f32,
    for_patient: bool,
) -> f32 {
    draw_section_header(layer, font_bold, "Imaging", start_y);
    let box_top = start_y - 12.0;

    let mut mri_drawn = false;
    let mut kspace_drawn = false;

    if for_patient {
        if let Some(mri) = mri_path {
            if draw_image_at(layer, mri, 172.5, box_top, 250.0, 250.0).is_ok() {
                draw_text_line(layer, "Reconstructed MRI Slice (Middle)", font_bold, 9.5, 212.5, box_top - 262.0, 4.0/255.0, 44.0/255.0, 83.0/255.0);
                mri_drawn = true;
            }
        }
        if !mri_drawn {
            draw_rect_filled(layer, 172.5, box_top - 250.0, 250.0, 250.0, 240.0/255.0, 244.0/255.0, 248.0/255.0);
            draw_text_line(layer, "[MRI Image Missing]", font_bold, 10.0, 242.5, box_top - 125.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
        }
    } else {
        if let Some(mri) = mri_path {
            if draw_image_at(layer, mri, 40.0, box_top, 250.0, 250.0).is_ok() {
                draw_text_line(layer, "Reconstructed MRI Slice (Middle)", font_bold, 9.5, 40.0, box_top - 262.0, 4.0/255.0, 44.0/255.0, 83.0/255.0);
                mri_drawn = true;
            }
        }

        if let Some(kspace) = kspace_path {
            if draw_image_at(layer, kspace, 305.0, box_top, 250.0, 250.0).is_ok() {
                draw_text_line(layer, "K-Space Log-Magnitude (Middle)", font_bold, 9.5, 305.0, box_top - 262.0, 4.0/255.0, 44.0/255.0, 83.0/255.0);
                kspace_drawn = true;
            }
        }

        if !mri_drawn && kspace_drawn {
            draw_rect_filled(layer, 40.0, box_top - 250.0, 250.0, 250.0, 240.0/255.0, 244.0/255.0, 248.0/255.0);
            draw_text_line(layer, "[MRI Image Missing]", font_bold, 10.0, 110.0, box_top - 125.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
        }

        if mri_drawn && !kspace_drawn {
            draw_rect_filled(layer, 305.0, box_top - 250.0, 250.0, 250.0, 240.0/255.0, 244.0/255.0, 248.0/255.0);
            draw_text_line(layer, "[K-Space Image Missing]", font_bold, 10.0, 375.0, box_top - 125.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
        }
    }

    if mri_drawn || kspace_drawn {
        box_top - 280.0
    } else {
        let placeholder_height = 140.0;
        let placeholder_bottom = box_top - placeholder_height;
        draw_rect_filled(layer, 40.0, placeholder_bottom, 515.0, placeholder_height, 230.0/255.0, 241.0/255.0, 251.0/255.0);

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
            pt(40.0, placeholder_bottom),
            pt(555.0, placeholder_bottom),
            pt(555.0, box_top),
            pt(40.0, box_top),
        ];
        layer.add_polygon(Polygon {
            rings: vec![border_points],
            mode: printpdf::path::PaintMode::Stroke,
            winding_order: printpdf::path::WindingOrder::NonZero,
        });
        
        layer.set_line_dash_pattern(LineDashPattern {
            offset: 0,
            dash_1: None,
            gap_1: None,
            dash_2: None,
            gap_2: None,
            dash_3: None,
            gap_3: None,
        });

        draw_text_line(layer, "[o]", font_bold, 20.0, 280.0, box_top - 50.0, 55.0/255.0, 138.0/255.0, 221.0/255.0);
        draw_text_line(layer, "MRI scan image", font_bold, 12.0, 245.0, box_top - 80.0, 24.0/255.0, 95.0/255.0, 165.0/255.0);
        draw_text_line(layer, "Image will be inserted here", font_regular, 11.0, 215.0, box_top - 105.0, 55.0/255.0, 138.0/255.0, 221.0/255.0);

        placeholder_bottom
    }
}

fn draw_impression_section(
    layer: &PdfLayerReference,
    font_regular: &IndirectFontRef,
    font_bold: &IndirectFontRef,
    items: &[String],
    start_y: f32,
    max_width: f32,
    label: &str,
    for_patient: bool,
) -> f32 {
    draw_section_header(layer, font_bold, label, start_y);
    let box_top = start_y - 12.0;

    let line_height = 16.0;
    let text_margin = 12.0;
    let list_x = 60.0;
    let list_max_width = max_width - 30.0;
    
    let mut wrapped_items = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let full_text = if for_patient {
            item.to_string()
        } else {
            format!("{}. {}", i + 1, item)
        };
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
            draw_rich_text_line(layer, &line, font_regular, font_bold, 13.0, list_x, current_y, 4.0/255.0, 44.0/255.0, 83.0/255.0);
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
        draw_text_line(&layer, "KVISION // Clinical Imaging Report (Continued)", &self.font_bold, 11.0, 55.0, self.y_cursor - 25.0, 1.0, 1.0, 1.0);
        
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
    let mut mri_override: Option<String> = None;
    let mut kspace_override: Option<String> = None;
    let mut patient_id_override: Option<String> = None;
    let mut study_date_override: Option<String> = None;
    let mut modality_override: Option<String> = None;
    let mut for_patient = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--for-patient" => {
                for_patient = true;
                i += 1;
            }
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
            "--mri" => {
                if i + 1 < args.len() {
                    mri_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--kspace" => {
                if i + 1 < args.len() {
                    kspace_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--patient-id" => {
                if i + 1 < args.len() {
                    patient_id_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--study-date" => {
                if i + 1 < args.len() {
                    study_date_override = Some(args[i + 1].clone());
                    i += 2;
                }
            }
            "--modality" => {
                if i + 1 < args.len() {
                    modality_override = Some(args[i + 1].clone());
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
    if let Some(pid) = patient_id_override { report.patient_id = pid; }
    if let Some(sd) = study_date_override { report.study_date = sd; }
    if let Some(m) = modality_override { report.modality = m; }

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
        // Left text:
        draw_text_line(&layer, "KVISION // CLINICAL IMAGING CENTER", &state.font_bold, 14.0, 40.0, 775.0, 12.0/255.0, 68.0/255.0, 124.0/255.0);
        draw_text_line(&layer, "MAGNETOM TRIO 3T MRI WORKSTATION", &state.font_regular, 8.5, 40.0, 760.0, 74.0/255.0, 85.0/255.0, 104.0/255.0);

        // Right text:
        draw_text_line(&layer, "123 Health Ave, Suite 400", &state.font_regular, 8.5, 455.0, 778.0, 74.0/255.0, 85.0/255.0, 104.0/255.0);
        draw_text_line(&layer, "Phone: (555) 019-2834", &state.font_regular, 8.5, 471.0, 766.0, 74.0/255.0, 85.0/255.0, 104.0/255.0);
        draw_text_line(&layer, "reports@kvision.ai", &state.font_regular, 8.5, 485.0, 754.0, 74.0/255.0, 85.0/255.0, 104.0/255.0);

        // Thin separator line
        draw_line(&layer, 40.0, 746.0, 555.0, 746.0, 1.0, 12.0/255.0, 68.0/255.0, 124.0/255.0);
    }
    
    // 2. Draw Patient Info Table on Page 1
    {
        let layer = state.current_layer();
        let table_top = 730.0;
        let row_height = 20.0;

        // Fill background for Label columns
        for r in 0..4 {
            let y_top = table_top - (r as f32) * row_height;
            let y_bottom = y_top - row_height;
            // Col 1 label
            draw_rect_filled(&layer, 40.0, y_bottom, 95.0, row_height, 240.0/255.0, 244.0/255.0, 248.0/255.0);
            // Col 2 label
            draw_rect_filled(&layer, 285.0, y_bottom, 110.0, row_height, 240.0/255.0, 244.0/255.0, 248.0/255.0);
        }

        // Draw horizontal grid lines
        for r in 0..=4 {
            let y = table_top - (r as f32) * row_height;
            draw_line(&layer, 40.0, y, 555.0, y, 0.75, 200.0/255.0, 210.0/255.0, 220.0/255.0);
        }

        // Draw vertical grid lines
        let x_coords = [40.0, 135.0, 285.0, 395.0, 555.0];
        for x in x_coords {
            draw_line(&layer, x, table_top - 80.0, x, table_top, 0.75, 200.0/255.0, 210.0/255.0, 220.0/255.0);
        }

        // Column value-cell widths (available space between value start and next border)
        // Col-1 value: 141 → 285  = 144pt; Col-2 value: 401 → 555 = 154pt
        let col1_val_w = 138.0_f32;  // 141..285 with 6pt right padding
        let col2_val_w = 148.0_f32;  // 401..555 with 6pt right padding

        // Row 0
        let y0 = table_top - 1.0 * row_height + 5.5;
        draw_text_line(&layer, "Patient Name:", &state.font_bold, 9.0, 46.0, y0, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, &truncate_text(&report.patient_name, col1_val_w, 9.0), &state.font_regular, 9.0, 141.0, y0, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, "Referring Physician:", &state.font_bold, 9.0, 291.0, y0, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, &truncate_text(&report.referring_physician, col2_val_w, 9.0), &state.font_regular, 9.0, 401.0, y0, 4.0/255.0, 44.0/255.0, 83.0/255.0);

        // Row 1
        let y1 = table_top - 2.0 * row_height + 5.5;
        draw_text_line(&layer, "Patient ID:", &state.font_bold, 9.0, 46.0, y1, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, &truncate_text(&report.patient_id, col1_val_w, 9.0), &state.font_regular, 9.0, 141.0, y1, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, "Modality:", &state.font_bold, 9.0, 291.0, y1, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, &truncate_text(&report.modality, col2_val_w, 9.0), &state.font_regular, 9.0, 401.0, y1, 4.0/255.0, 44.0/255.0, 83.0/255.0);

        // Row 2
        let y2 = table_top - 3.0 * row_height + 5.5;
        draw_text_line(&layer, "Age / Gender:", &state.font_bold, 9.0, 46.0, y2, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        let age_sex = format!("{} / {}", report.patient_age, report.patient_sex);
        draw_text_line(&layer, &truncate_text(&age_sex, col1_val_w, 9.0), &state.font_regular, 9.0, 141.0, y2, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, "Date of Study:", &state.font_bold, 9.0, 291.0, y2, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, &truncate_text(&report.study_date, col2_val_w, 9.0), &state.font_regular, 9.0, 401.0, y2, 4.0/255.0, 44.0/255.0, 83.0/255.0);

        // Row 3
        let y3 = table_top - 4.0 * row_height + 5.5;
        draw_text_line(&layer, "Report ID:", &state.font_bold, 9.0, 46.0, y3, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, &truncate_text(&report.report_id, col1_val_w, 9.0), &state.font_regular, 9.0, 141.0, y3, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, "Date of Report:", &state.font_bold, 9.0, 291.0, y3, 4.0/255.0, 44.0/255.0, 83.0/255.0);
        draw_text_line(&layer, &truncate_text(&report.date, col2_val_w, 9.0), &state.font_regular, 9.0, 401.0, y3, 4.0/255.0, 44.0/255.0, 83.0/255.0);
    }
    
    // 3. Draw centered bold title
    {
        let layer = state.current_layer();
        // Above line
        draw_line(&layer, 40.0, 634.0, 555.0, 634.0, 0.5, 200.0/255.0, 210.0/255.0, 220.0/255.0);
        // Title Text
        draw_text_line(&layer, "MAGNETIC RESONANCE IMAGING (MRI) BRAIN REPORT", &state.font_bold, 11.0, 160.0, 621.0, 12.0/255.0, 68.0/255.0, 124.0/255.0);
        // Below line
        draw_line(&layer, 40.0, 614.0, 555.0, 614.0, 0.5, 200.0/255.0, 210.0/255.0, 220.0/255.0);
    }

    state.y_cursor = 595.0;

    // 3. Draw BODY SECTIONS dynamically
    
    let label_clinical = if for_patient { "Patient MRI Summary" } else { "Clinical Indication" };
    let label_technique = if for_patient { "Explanation of Tech" } else { "Technique" };
    let label_findings = if for_patient { "Detailed Explanation" } else { "Findings" };
    let label_impression = if for_patient { "What Was Found" } else { "Impression" };
    let label_recommendation = if for_patient { "Next Steps & Recommendations" } else { "Recommendation" };

    // a. Clinical indication
    draw_standard_section(&mut state, &report, label_clinical, &report.clinical_indication, 515.0);
    state.y_cursor -= 15.0;

    // b. Technique
    draw_standard_section(&mut state, &report, label_technique, &report.technique, 515.0);
    state.y_cursor -= 15.0;

    // c. Imaging
    let height_c = if mri_override.is_some() || kspace_override.is_some() { 285.0 } else { 165.0 };
    state.ensure_space(height_c, &report);
    state.y_cursor = draw_imaging_section(
        &state.current_layer(),
        &state.font_regular,
        &state.font_bold,
        mri_override.as_deref(),
        kspace_override.as_deref(),
        state.y_cursor,
        515.0,
        for_patient,
    ) - 15.0;

    // d. Findings
    draw_standard_section(&mut state, &report, label_findings, &report.findings, 515.0);
    state.y_cursor -= 15.0;

    // e. Impression
    let mut total_lines = 0;
    for (i, item) in report.impression.iter().enumerate() {
        let full_text = if for_patient {
            item.to_string()
        } else {
            format!("{}. {}", i + 1, item)
        };
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
        label_impression,
        for_patient,
    ) - 15.0;

    // f. Recommendation
    draw_standard_section(&mut state, &report, label_recommendation, &report.recommendation, 515.0);
    state.y_cursor -= 15.0;

    // 4. Draw FOOTER on the last page
    draw_footer(&state.current_layer(), &state.font_regular);

    // Save document
    let file = File::create(&output_path)?;
    state.doc.save(&mut BufWriter::new(file))?;
    
    println!("Success: PDF report generated at {}", output_path);
    Ok(())
}
