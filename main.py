import argparse
from src.stream_analyzer import Stream_Analyzer
import time

import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
import OpenGL.GL.shaders
import numpy as np

ear = Stream_Analyzer(
    # Pyaudio (portaudio) device index, defaults to first mic input
    device=None,
    rate=None,               # Audio samplerate, None uses the default source settings
    FFT_window_size_ms=60,    # Window size used for the FFT transform
    updates_per_second=2000,  # How often to read the audio stream for new data
    smoothing_length_ms=50,    # Apply some temporal smoothing to reduce noisy features
    n_frequency_bins=300,  # The FFT features are grouped in bins
    visualize=0,               # Visualize the FFT features with PyGame
    verbose=0,    # Float ratio of the visualizer window. e.g. 24/9
)

fps = 60


def main():

    if not glfw.init():
        return

    window = glfw.create_window(1200, 1000, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()

    l = list(range(raw_fftx.size * 3))  # Make a list of 1000 None's
    for i in range(0, raw_fftx.size):
        l[3*i+0] = i * 0.0
        l[3*i+1] = 0.0
        l[3*i+2] = 0.0

    vertex = np.array(l, dtype=np.float32)

    strip = np.arange(raw_fftx.size, dtype=np.uintc)

    vertex_shader = """
    #version 330
    #define PI 3.1415926538

    //attribute vec3 position;
    attribute float frequency;
    attribute float spec;

    out vec3 newColor;
    out float newSpec;

    vec3 convertHSVToRGB(in float hue, in float saturation, in float lightness) {
        float chroma = lightness * saturation;
        float hueDash = hue / 60.0;
        float x = chroma * (1.0 - abs(mod(hueDash, 2.0) - 1.0));
        vec3 hsv = vec3(0.0);

        if(hueDash < 1.0) {
            hsv.r = chroma;
            hsv.g = x;
        } else if (hueDash < 2.0) {
            hsv.r = x;
            hsv.g = chroma;
        } else if (hueDash < 3.0) {
            hsv.g = chroma;
            hsv.b = x;
        } else if (hueDash < 4.0) {
            hsv.g = x;
            hsv.b = chroma;
        } else if (hueDash < 5.0) {
            hsv.r = x;
            hsv.b = chroma;
        } else if (hueDash < 6.0) {
            hsv.r = chroma;
            hsv.b = x;
        }

        return hsv;
    }

    void main()
    {
        float verticalScale = 1.0f;
        float C2 = 65.4f;
        vec3 pos = vec3(0.0f,0.0f,0.0f);
        if(frequency / C2 >= 1){
            float radius = frequency / C2 * 0.005f;
            float angle = 2 * PI * log2(frequency / C2);
            float s = spec;
            pos = radius * vec3(cos(angle), sin(angle), 0.0f);
        } 
        gl_Position = 0.0f * vec4(frequency/100, 0.0f,0.0f,1.0f) + 1.0f * vec4(pos , 1.0f);
        gl_PointSize = 10.0f; 

        float hue = 360.0 - ((spec / verticalScale) * 360.0);
        //newColor = convertHSVToRGB(hue, 1.0, 1.0);
        newSpec = spec;
        newColor = vec3(1.0f,1.0f,1.0f);
    }
    """
    fragment_shader = """
    #version 330

    in vec3 newColor;
    in float newSpec;

    out vec4 outColor;
    void main()
    {
        vec4 backgroundColor = vec4(0.0f,0.0f,0.0f,1.0f);
        float fade = pow(cos((1.0 - newSpec) * 0.5 * 3.1415926535), 0.5);

        outColor = vec4(newSpec * newColor,1.0f);
        // k *= fade;
        //outColor = backgroundColor + vec4(fade * newColor, 1.0);

        //outColor = vec4(newColor, 1.0f);
    }
    """
    glEnable(GL_PROGRAM_POINT_SIZE)

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 4 * vertex.size, vertex, GL_STATIC_DRAW)

    frequencyBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, frequencyBuffer)

    frequency = glGetAttribLocation(shader, "frequency")
    glVertexAttribPointer(frequency, 1, GL_FLOAT,
                          GL_FALSE, 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(frequency)

    spectrumBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, spectrumBuffer)

    spec = glGetAttribLocation(shader, "spec")
    glVertexAttribPointer(spec, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(spec)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 *
                 strip.size, strip, GL_STATIC_DRAW)

    glUseProgram(shader)

    glClearColor(0, 0, 0, 1.0)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT)

        raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()

        raw_fftx = raw_fftx.astype(np.float32)
        raw_fft = raw_fft.astype(np.float32)

        norm = np.linalg.norm(raw_fft)
        raw_fft = raw_fft/norm

        binned_fftx = binned_fftx.astype(np.float32)
        binned_fft = binned_fft.astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, frequencyBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * raw_fftx.size,
                     raw_fftx, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, spectrumBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * raw_fftx.size,
                     raw_fft, GL_STATIC_DRAW)

        glDrawElements(GL_LINE_STRIP, strip.size, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
