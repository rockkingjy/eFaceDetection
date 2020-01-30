#include "common.h"
/* 
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// -------------- image.c --------------
void rgbgr_image(image im)
{
    int i;
    for (i = 0; i < im.w*im.h; ++i) {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w*im.h * 2];
        im.data[i + im.w*im.h * 2] = swap;
    }
}

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    int i;
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (i = x1; i <= x2; ++i) {
        a.data[i + y1*a.w + 0 * a.w*a.h] = r;
        a.data[i + y2*a.w + 0 * a.w*a.h] = r;

        a.data[i + y1*a.w + 1 * a.w*a.h] = g;
        a.data[i + y2*a.w + 1 * a.w*a.h] = g;

        a.data[i + y1*a.w + 2 * a.w*a.h] = b;
        a.data[i + y2*a.w + 2 * a.w*a.h] = b;
    }
    for (i = y1; i <= y2; ++i) {
        a.data[x1 + i*a.w + 0 * a.w*a.h] = r;
        a.data[x2 + i*a.w + 0 * a.w*a.h] = r;

        a.data[x1 + i*a.w + 1 * a.w*a.h] = g;
        a.data[x2 + i*a.w + 1 * a.w*a.h] = g;

        a.data[x1 + i*a.w + 2 * a.w*a.h] = b;
        a.data[x2 + i*a.w + 2 * a.w*a.h] = b;
    }
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for (i = 0; i < w; ++i) {
        draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
    }
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c*w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r*h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_in_out(&part);
    return resized;
}

image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i, j, k;
    image im = make_image(w, h, c);
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index];// / 255.;
            }
        }
    }
    free(data);
    return im;
}

image load_image(char *filename, int w, int h, int c)
{
    image out = load_image_stb(filename, c);
    if ((h && w) && (h != out.h || w != out.w)) {
        image resized = resize_image(out, w, h);
        free_in_out(&out);
        out = resized;
    }
    return out;
}

*/
image load_image(char *filename, int w, int h, int c)
{
    image out = {0, 0, 0, NULL};
    out.w = w;
    out.h = h;
    out.c = c;
    out.data = calloc(w * h * c, sizeof(float));
    FILE *file = fopen(filename, "rb");
    if (file == 0)
    {
        printf("Couldn't open file: %s\n", filename);
        out.w = 0;
        out.h = 0;
        out.c = 0;
        free(out.data);
        return out;
    }

    for (int k = 0; k < c; k++)
        for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++)
            {
                out.data[k * w * h + j * w + i] = fgetc(file);
            }
    fclose(file);
    return out;
}

void free_in_out(image *m)
{
    if (m->data)
    {
        free(m->data);
        m->data = NULL;
    }
}

void print_in_out(image im)
{
    for (int k = 0; k < im.c; k++)
    {
        for (int j = 0; j < im.h; j++)
        {
            for (int i = 0; i < im.w; i++)
            {
                printf("%3f ", im.data[k * im.w * im.h + j * im.w + i]);
            }
            printf("\n");
        }
    }
}

void save_in_out(image im)
{
    char *filename = "./temp_save_layer_c.txt";
    FILE *file = fopen(filename, "w");
    for (int k = 0; k < im.c; k++)
    {
        for (int j = 0; j < im.h; j++)
        {
            for (int i = 0; i < im.w; i++)
            {
                fprintf(file, "%9f,", im.data[k * im.w * im.h + j * im.w + i]);
                //printf("%9f ", im.data[k * im.w * im.h + j * im.w + i]);
            }
            fprintf(file, "\n");
            //printf("\n");
        }
    }
    fclose(file);
}

void normalize_image(image *im, float mean, float std)
{
    for (int k = 0; k < im->c; k++)
    {
        for (int j = 0; j < im->h; j++)
        {
            for (int i = 0; i < im->w; i++)
            {
                im->data[k * im->w * im->h + j * im->w + i] -= mean;
                im->data[k * im->w * im->h + j * im->w + i] /= std;
            }
        }
    }
}

void normalize_image_255(image *im)
{
    for (int k = 0; k < im->c; k++)
    {
        for (int j = 0; j < im->h; j++)
        {
            for (int i = 0; i < im->w; i++)
            {
                im->data[k * im->w * im->h + j * im->w + i] /= 255.0f;
            }
        }
    }
}