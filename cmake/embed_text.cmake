# Writes the text of IN as a C++ raw string literal to OUT, so a shader source
# can be compiled into the binary and compiled by Metal at runtime.
file(READ "${IN}" content)
file(WRITE "${OUT}" "R\"BRISK_EMBED(${content})BRISK_EMBED\"")
