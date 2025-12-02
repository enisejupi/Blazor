"""
Ekzekuto projektin e plotë automatikisht
Run the complete project automatically
"""

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   PROJEKTI I KRAHASIMIT TË METODAVE TË HEQJES SË ZHURMËS           ║
║   IMAGE DENOISING METHODS COMPARISON PROJECT                        ║
║                                                                      ║
║   Zgjidhni një opsion / Choose an option:                          ║
║                                                                      ║
║   1. 🚀 Ekzekuto pipeline-in e plotë (Run full pipeline)           ║
║   2. ⚡ Ekzekuto pipeline-in e shpejtë (Quick run)                 ║
║   3. 📊 Hap dashboard-in (Open dashboard)                          ║
║   4. 📥 Shkarko vetëm imazhe (Download images only)                ║
║   5. ❌ Dil (Exit)                                                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

choice = input("Zgjedhja juaj / Your choice (1-5): ").strip()

if choice == '1':
    print("\n🚀 Duke ekzekutuar pipeline-in e plotë...")
    print("⏱️  Ky proces mund të zgjasë 2-4 orë (në varësi të hardware-it)")
    confirm = input("Vazhdoni? (y/n): ").strip().lower()
    if confirm == 'y':
        import subprocess
        subprocess.run(['python', 'run_pipeline.py'])
    else:
        print("❌ Anulluar")

elif choice == '2':
    print("\n⚡ Duke ekzekutuar pipeline-in e shpejtë...")
    print("⏱️  Ky proces do të zgjasë ~15-30 minuta")
    import subprocess
    subprocess.run(['python', 'run_pipeline.py', '--quick'])

elif choice == '3':
    print("\n📊 Duke hapur dashboard-in...")
    print("🌐 Dashboard-i do të hapet në shfletuesin tuaj")
    print("📍 URL: http://localhost:8501")
    print("\n💡 Për të mbyllur dashboard-in, shtypni Ctrl+C në terminal")
    import subprocess
    import sys
    
    try:
        # Try using python -m streamlit which works even if streamlit isn't in PATH
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard_app.py'])
    except FileNotFoundError:
        print("\n❌ Gabim: Streamlit nuk është instaluar / Error: Streamlit not installed")
        print("💡 Instaloni me: pip install streamlit")
        print("💡 Install with: pip install streamlit")
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard-i u mbyll / Dashboard closed")
    except Exception as e:
        print(f"\n❌ Gabim gjatë hapjes së dashboard-it / Error opening dashboard: {e}")

elif choice == '4':
    print("\n📥 Duke shkarkuar imazhe...")
    from src.data_loader import ImageDownloader
    downloader = ImageDownloader()
    downloader.download_images()
    print("\n✅ Imazhet u shkarkuan me sukses!")
    print(f"📁 Vendndodhja: data/images/")

elif choice == '5':
    print("\n👋 Mirupafshim! / Goodbye!")

else:
    print("\n❌ Zgjedhje e pavlefshme / Invalid choice")
    print("Ju lutem ekzekutoni skriptin përsëri dhe zgjidhni 1-5")

print("\n" + "="*70)
