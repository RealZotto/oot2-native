// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace Oot2.Xlate;

/// <summary>
/// oot2-xlate: translates Switch 2 (NVN2, Ampere sm_86) shader programs to
/// SPIR-V for Vulkan 1.3. See docs/architecture.md, D20 and D21, and the job
/// contract in tools/shaders/pool.py.
///
///   oot2-xlate --serve      read one JSON request per stdin line, answer on stdout
///   oot2-xlate --version    print the translator identity and exit
///   oot2-xlate --fixup-module <in.spv> <out.spv>
///                           run the post-pass on a SPIR-V file and print the capabilities
///                           it added, one per line, then the decorations it added, as
///                           NonWritable@set.binding or NonReadable@set.binding (a test aid:
///                           the post-pass rules can then be exercised without a shader that
///                           triggers them)
///
/// The identity is the SHA-256 of this executable. Shader cache keys include it
/// (D21), so any change to the translator or to this host invalidates the cache.
/// </summary>
internal static class Program
{
    // A deeply nested shader recurses far in the structured-IR passes.
    private const int ServeStackBytes = 256 * 1024 * 1024;

    private static int Main(string[] args)
    {
        if (args.Length == 1 && args[0] == "--version")
        {
            // optimizer= names the form a job gets when it sets no `debug` option: "sound" is the AST
            // optimizer rule of AstOptimizer.cs, "debug" the old form. prepare refuses a translator whose
            // default is not the sound one, so a pack never holds modules of the other form.
            string optimizer = new JobOptions().Debug ? "debug" : "sound";

            Console.Out.WriteLine($"oot2-xlate {ExecutableSha256()} proto={Host.Protocol} spirv=1.3 target=vulkan1.3 optimizer={optimizer}");

            return 0;
        }

        if (args.Length == 1 && args[0] == "--serve")
        {
            // The replies go to the raw stdout stream. Whatever a library prints
            // through Console.Out lands on stderr instead of in the protocol.
            Stream replies = Console.OpenStandardOutput();
            Console.SetOut(Console.Error);

            int exit = 0;
            Thread thread = new(() => exit = Serve(Console.OpenStandardInput(), replies), ServeStackBytes);
            thread.Start();
            thread.Join();

            return exit;
        }

        if (args.Length == 3 && args[0] == "--fixup-module")
        {
            return FixupFile(args[1], args[2]);
        }

        Console.Error.WriteLine("usage: oot2-xlate --serve | --version | --fixup-module <in.spv> <out.spv>");

        return 2;
    }

    private static int FixupFile(string input, string output)
    {
        try
        {
            FixupResult result = SpirvFixup.Apply(File.ReadAllBytes(input), true);
            File.WriteAllBytes(output, result.Module);

            foreach (uint capability in result.Added)
            {
                Console.Out.WriteLine(SpirvFixup.CapabilityName(capability));
            }

            foreach (string decoration in result.Decorated)
            {
                Console.Out.WriteLine(decoration);
            }

            return 0;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"fixup-module: {e.Message}");

            return 1;
        }
    }

    private static int Serve(Stream input, Stream output)
    {
        Host host = new(ExecutableSha256());
        using StreamReader reader = new(input, new UTF8Encoding(false), false, 1 << 16);

        while (reader.ReadLine() is { } line)
        {
            string reply;

            try
            {
                reply = host.Handle(line);
            }
            catch (Exception e)
            {
                // Handle classifies everything itself; this is the last guard
                // against a reply line never being written.
                reply = $"fail - 0 {Host.Sanitize($"translate: {e.GetType().Name}: {e.Message}", 400)}";
            }

            if (reply != null)
            {
                byte[] bytes = Encoding.UTF8.GetBytes(reply + "\n");
                output.Write(bytes, 0, bytes.Length);
                output.Flush();
            }

            if (host.Quit)
            {
                break;
            }
        }

        return 0;
    }

    private static string ExecutableSha256()
    {
        string path = Environment.ProcessPath;

        if (path == null || !File.Exists(path))
        {
            return new string('0', 64);
        }

        using FileStream fs = File.OpenRead(path);

        return Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
    }
}
