#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

unsigned char *read_pgm(const char *filename, int *width, int *height)
{
    FILE *f = fopen(filename, "rb");

    char magic[3];
    fscanf(f, "%2s", magic);

    int maxval;
    fscanf(f, "%d %d %d", width, height, &maxval);
    fgetc(f);

    unsigned char *data = (unsigned char *)malloc(*width * *height);
    fread(data, 1, *width * *height, f);
    fclose(f);
    return data;
}

void apply_sobel(unsigned char *img, int *mask, unsigned char *res,
                 int width, int height, int mw, int mh)
{
    int hmw = mw / 2;
    int hmh = mh / 2;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int sum = 0;
            for (int j = -hmh; j <= hmh; j++)
            {
                for (int i = -hmw; i <= hmw; i++)
                {
                    int xi = x + i;
                    int yi = y + j;
                    if (xi >= 0 && xi < width && yi >= 0 && yi < height)
                    {
                        int mask_idx = (j + hmh) * mw + (i + hmw);
                        sum += img[yi * width + xi] * mask[mask_idx];
                    }
                }
            }
            sum = (sum < 0) ? 0 : (sum > 255) ? 255
                                              : sum;
            res[y * width + x] = (unsigned char)sum;
        }
    }
}

int main()
{
    const char *fname = "Mona_Lisa.pgm";
    int mask[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int width, height;

    clock_t start = clock();
    unsigned char *img = read_pgm(fname, &width, &height);
    unsigned char *res = malloc(width * height);

    apply_sobel(img, mask, res, width, height, 3, 3);

    FILE *out = fopen("Mona_Lisa_out.pgm", "wb");
    fprintf(out, "P5\n%d %d\n255\n", width, height);
    fwrite(res, 1, width * height, out);
    fclose(out);

    clock_t end = clock();
    printf("CPU Execution time: %.2f ms\n",
           ((double)(end - start)) * 1000 / CLOCKS_PER_SEC);

    free(img);
    free(res);
    return 0;
}
