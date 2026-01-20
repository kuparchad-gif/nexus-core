import os, shutil, pathlib
import typer

app = typer.Typer(help="Lillith Firmware Foundry — scaffold edge firmware projects.")

ROOT = pathlib.Path(__file__).resolve().parents[1]
TPL = ROOT / "templates"
LIB = ROOT / "libs"

def copy_tree(src: pathlib.Path, dst: pathlib.Path):
    if not src.exists():
        raise RuntimeError(f"Missing template path: {src}")
    for root, dirs, files in os.walk(src):
        rel = pathlib.Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            s = pathlib.Path(root) / f
            d = (dst / rel / f)
            shutil.copy2(s, d)

@app.command()
def forge(target: str = typer.Argument(..., help="micropython | esp-idf"),
          name: str = typer.Option("app", help="Project folder name"),
          router: str = typer.Option("[REDACTED-URL] help="Router URL"),
          sensor: str = typer.Option("bh1750", help="Sensor type (micropython: bh1750)"),
          wifi_ssid: str = typer.Option("LILLITH_NET"),
[REDACTED-SECRET-LINE]
    out = pathlib.Path.cwd() / name
    out.mkdir(parents=True, exist_ok=True)

    if target == "micropython":
        copy_tree(TPL/"micropython", out)
        cfg = out/"net.py"
        text = cfg.read_text(encoding="utf-8")
[REDACTED-SECRET-LINE]
        cfg.write_text(text, encoding="utf-8")
        gate = out/"gate.py"
        gate.write_text(gate.read_text(encoding="utf-8").replace("{{ROUTER_URL}}", router), encoding="utf-8")
        typer.echo(f"Micropython project created at {out}")
        return

    if target == "esp-idf":
        copy_tree(TPL/"esp-idf", out)
        g = out/"main"/"gate_client.c"
        text = g.read_text(encoding="utf-8").replace("{{ROUTER_URL}}", router)
        g.write_text(text, encoding="utf-8")
        typer.echo(f"ESP-IDF project created at {out}")
        return

    raise typer.BadParameter("Unknown target. Use 'micropython' or 'esp-idf'.")

if __name__ == "__main__":
    app()

