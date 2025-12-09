import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Final Submission - Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def write_formatted_line(self, text, prefix=''):
        if prefix:
            self.set_font('Arial', '', 11)
            self.write(6, prefix)
            
        parts = text.split('**')
        for i, part in enumerate(parts):
            # Clean up LaTeX markers
            part = part.replace('$H_0$', 'H0').replace('$', '')
            
            if i % 2 == 1:
                self.set_font('Arial', 'B', 11)
            else:
                self.set_font('Arial', '', 11)
            self.write(6, part)
        self.ln()

    def render_table(self, rows):
        # rows is a list of lists (columns)
        if not rows:
            return

        # Determine column widths dynamically
        num_cols = len(rows[0])
        page_width = 190
        
        # Simple heuristic: Distribute evenly, or use specific ratios if known
        # For this report, we have 3-col and 4-col tables.
        if num_cols == 3:
            col_widths = [60, 65, 65]
        elif num_cols == 4:
            col_widths = [45, 35, 25, 85]
        else:
            col_widths = [page_width / num_cols] * num_cols
        
        # Header
        self.set_font('Arial', 'B', 10)
        header = rows[0]
        self.render_table_row(header, col_widths, is_header=True)
        
        # Data
        self.set_font('Arial', '', 10)
        for row in rows[1:]:
            self.render_table_row(row, col_widths)
        
        self.ln(5)

    def render_table_row(self, row_cols, col_widths, is_header=False):
        start_x = self.get_x()
        start_y = self.get_y()
        
        # Check page break
        if start_y > 250:
            self.add_page()
            start_y = self.get_y()
            start_x = self.get_x() # Reset X
            
        max_y = start_y
        
        current_x = start_x
        
        for i, text in enumerate(row_cols):
            if i >= len(col_widths): break
            w = col_widths[i]
            
            self.set_xy(current_x, start_y)
            
            # Formatting
            clean_text = text.strip().replace('$H_0$', 'H0').replace('$', '')
            is_bold = False
            if clean_text.startswith('**') and clean_text.endswith('**'):
                is_bold = True
                clean_text = clean_text[2:-2]
            
            if is_header or is_bold:
                self.set_font('Arial', 'B', 10)
            else:
                self.set_font('Arial', '', 10)
                
            # MultiCell
            self.multi_cell(w, 5, clean_text, border=1, align='L')
            
            if self.get_y() > max_y:
                max_y = self.get_y()
            
            current_x += w
            
        # Move to next line
        self.set_xy(start_x, max_y)

    def add_markdown_content(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        table_buffer = []
        
        for line in lines:
            line = line.strip()
            
            # Table detection
            if '|' in line:
                if '---' in line: continue # Skip separator
                cols = [c.strip() for c in line.split('|') if c.strip()]
                if cols:
                    table_buffer.append(cols)
                continue
            else:
                # If we have a buffer, render it now
                if table_buffer:
                    self.render_table(table_buffer)
                    table_buffer = []
            
            if not line:
                self.ln(2) # Add a small gap for empty lines
                continue
            
            if line.startswith('## '):
                self.chapter_title(line[3:])
            elif line.startswith('# '):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, line[2:], 0, 1, 'C')
                self.ln(5)
            elif line.startswith('### '):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 8, line[4:], 0, 1, 'L')
            elif line.startswith('- '):
                # Handle bullet points with potential bold text
                self.write_formatted_line(line[2:], prefix=chr(149) + ' ')
            elif line.startswith('**') and line.endswith('**'):
                 # Whole line bold
                 self.set_font('Arial', 'B', 11)
                 self.multi_cell(0, 6, line.replace('**', ''))
            elif line.startswith('![') and '](' in line and line.endswith(')'):
                # Image handling: ![Alt text](path/to/image)
                try:
                    start_idx = line.find('](') + 2
                    end_idx = line.find(')', start_idx)
                    img_path = line[start_idx:end_idx]
                    
                    # Resolve relative path
                    base_dir = os.path.dirname(filepath)
                    full_img_path = os.path.join(base_dir, img_path)
                    
                    if os.path.exists(full_img_path):
                        # Center image and scale to fit width
                        self.ln(2)
                        # Get image dimensions to scale proportionally if needed
                        # For simplicity, just fit to page width - margins (190)
                        self.image(full_img_path, x=self.get_x(), w=170) 
                        self.ln(2)
                    else:
                        self.set_font('Arial', 'I', 10)
                        self.cell(0, 10, f"[Image not found: {img_path}]", 0, 1)
                except Exception as e:
                    print(f"Error adding image: {e}")
            else:
                # Normal paragraph with potential bold text
                self.write_formatted_line(line)
        
        # Flush table buffer if any at end
        if table_buffer:
            self.render_table(table_buffer)

def generate_report():
    pdf = PDF()
    pdf.add_page()
    
    report_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'final_submission.md')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'final_submission.pdf')
    
    if os.path.exists(report_path):
        pdf.add_markdown_content(report_path)
        pdf.output(output_path)
        print(f"PDF generated successfully: {output_path}")
    else:
        print(f"Report file not found: {report_path}")

if __name__ == '__main__':
    generate_report()
