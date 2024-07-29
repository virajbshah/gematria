int main() {
  // TODO(virajbshah): Make this do something meaningful.
  asm volatile(
    "        movl    $10, %ecx\n"
    "        xorl    %eax, %eax\n"
    "loop:\n"
    "        movl    %ecx, %edx\n"
    "        imull   %edx, %edx\n"
    "        addl    %edx, %eax\n"
    "        decl    %ecx\n"
    "        jnz     loop"
  );
  return 0;
}
