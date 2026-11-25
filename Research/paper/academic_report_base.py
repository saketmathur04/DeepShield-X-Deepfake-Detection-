import os
import sys
from fpdf import FPDF
import fitz  # PyMuPDF

class AcademicReport(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(auto=False)
        self.set_margins(16, 14, 16)  # 16mm margins -> printable width = 178mm
        self.usable_w = 178
        
        # Colors
        self.c_navy = (16, 44, 76)        # Deep Navy
        self.c_slate = (45, 85, 125)      # Slate Blue
        self.c_grey = (85, 95, 110)       # Dark Grey
        self.c_text = (30, 35, 45)        # Body text
        self.c_border = (210, 220, 232)   # Border Grey
        self.c_bg_light = (245, 248, 252) # Light Card Fill
        self.c_bg_subtle = (250, 252, 255)# Very Light Fill

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'B', 8)
            self.set_text_color(*self.c_navy)
            self.cell(108, 4.5, 'DEEPFAKE DETECTION SYSTEM (DEEPSHIELD-X)', align='L')
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(*self.c_grey)
            self.cell(70, 4.5, 'Academic B.Tech Major Project Report', align='R')
            self.ln(4.5)
            self.set_draw_color(*self.c_border)
            self.set_line_width(0.3)
            self.line(16, self.get_y(), 194, self.get_y())
            self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.3)
        self.line(16, self.get_y(), 194, self.get_y())
        self.ln(1.5)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*self.c_grey)
        self.cell(90, 4, 'Department of Computer Science & Engineering', align='L')
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(*self.c_navy)
        self.cell(88, 4, f'Page {self.page_no()} of 6', align='R')

    def sec_heading(self, number, title):
        self.ln(1.5)
        self.set_font('Helvetica', 'B', 10.5)
        self.set_text_color(*self.c_navy)
        self.cell(self.usable_w, 5.5, f"{number}. {title.upper()}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.c_navy)
        self.set_line_width(0.5)
        self.line(16, self.get_y(), 194, self.get_y())
        self.ln(2)

    def subsec_heading(self, title):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*self.c_slate)
        self.cell(self.usable_w, 4.8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.8)

    def p(self, text, h=4.2):
        self.set_font('Helvetica', '', 8.5)
        self.set_text_color(*self.c_text)
        self.multi_cell(self.usable_w, h, text)
        self.ln(1.5)

    def bold_bullet_block(self, title, text, h=4.0):
        """Clean structured block: bold title line followed by full-width text."""
        self.set_font('Helvetica', 'B', 8.5)
        self.set_text_color(*self.c_navy)
        self.cell(self.usable_w, 4.4, title, new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 8.3)
        self.set_text_color(*self.c_text)
        self.multi_cell(self.usable_w, h, text)
        self.ln(1.5)

    def callout_card(self, title, text, bg_color=None, border_color=None, title_color=None):
        """Clean boxed callout with subtle background and left accent bar."""
        if bg_color is None:
            bg_color = self.c_bg_light
        if border_color is None:
            border_color = self.c_slate
        if title_color is None:
            title_color = self.c_navy

        start_y = self.get_y()
        self.set_font('Helvetica', '', 8.0)
        lines = self.multi_cell(self.usable_w - 6, 3.8, text, output="LINES")
        box_h = 5.5 + (len(lines) * 3.8) + 3

        # Background fill & thin outline
        self.set_fill_color(*bg_color)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.25)
        self.rect(16, start_y, self.usable_w, box_h, 'DF')

        # Left thick accent bar
        self.set_draw_color(*border_color)
        self.set_line_width(1.2)
        self.line(16, start_y, 16, start_y + box_h)

        # Content
        self.set_xy(19, start_y + 1.8)
        self.set_font('Helvetica', 'B', 8.5)
        self.set_text_color(*title_color)
        self.cell(self.usable_w - 6, 4.2, title, new_x="LMARGIN", new_y="NEXT")
        self.set_x(19)
        self.set_font('Helvetica', '', 8.0)
        self.set_text_color(*self.c_text)
        self.multi_cell(self.usable_w - 6, 3.8, text)
        self.set_xy(16, start_y + box_h + 2)

    def draw_grid_table(self, headers, rows, col_widths, col_aligns=None, font_size=7.5, row_h=4.6, header_bg=None, champ_idx=None, bad_idx=None):
        """Draws a clean, professional grid table without cell clipping."""
        if col_aligns is None:
            col_aligns = ['C'] * len(headers)
        if header_bg is None:
            header_bg = self.c_navy

        # Header Row
        self.set_fill_color(*header_bg)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.25)
        self.set_font('Helvetica', 'B', font_size)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], row_h + 0.5, h, border=1, fill=True, align='C')
        self.ln()

        # Data Rows
        for r_idx, row in enumerate(rows):
            is_champ = (champ_idx is not None and r_idx == champ_idx)
            is_bad = (bad_idx is not None and r_idx == bad_idx)

            if is_champ:
                self.set_fill_color(230, 242, 255)
                self.set_font('Helvetica', 'B', font_size)
                self.set_text_color(*self.c_navy)
            elif is_bad:
                self.set_fill_color(255, 238, 238)
                self.set_font('Helvetica', 'B', font_size)
                self.set_text_color(165, 30, 30)
            else:
                if r_idx % 2 == 0:
                    self.set_fill_color(255, 255, 255)
                else:
                    self.set_fill_color(246, 249, 253)
                self.set_font('Helvetica', '', font_size)
                self.set_text_color(*self.c_text)

            for i, val in enumerate(row):
                align = col_aligns[i] if i < len(col_aligns) else 'C'
                self.cell(col_widths[i], row_h, str(val), border=1, fill=True, align=align)
            self.ln()
        self.ln(1.5)

print("AcademicReport base class defined.")
