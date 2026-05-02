import json
import sys
from playwright.sync_api import sync_playwright

def extraer_pagina12():
    data = []
    urls = [
        "https://www.pagina12.com.ar/2026/02/13/todos-los-derechos-con-los-que-arrasa-la-reforma-laboral-de-milei/",
        "https://www.pagina12.com.ar/2026/03/06/milei-promulgo-su-reforma-laboral-uno-por-uno-todos-los-derechos-que-pierden-los-trabajadores/",
        "https://www.pagina12.com.ar/2026/02/21/el-futuro-del-trabajo-con-la-reforma-laboral/",
        "https://www.pagina12.com.ar/2026/03/06/el-impacto-de-la-reforma-laboral/",
        "https://www.pagina12.com.ar/2026/01/25/la-reforma-laboral-de-milei-trae-mas-ajuste/",
        "https://www.pagina12.com.ar/2026/02/21/si-sale-la-reforma-laboral-la-pelea-sigue-en-los-tribunales/"
    ]

    # Usamos Raw String para evitar problemas de escape en Windows
    JS_EXTRACTOR = r"""
    () => {
        const getTxt = (sel) => document.querySelector(sel)?.innerText.trim() || "";
        let titulo = getTxt('h1') || getTxt('.article-title');
        let autor = getTxt('.article-author') || getTxt('.author-name') || "Redacción Página|12";

        let fecha = "Fecha no encontrada";
        const fechaEl = document.querySelector('time');
        if (fechaEl) {
            fecha = fechaEl.getAttribute('datetime') || fechaEl.innerText.trim();
        }

        let contenedorTexto = document.querySelector('.article-main-content') || 
                             document.querySelector('.article-text') || 
                             document.querySelector('article');

        let texto = "";
        if (contenedorTexto) {
            const parrafos = Array.from(contenedorTexto.querySelectorAll('p'));
            texto = parrafos
                .map(p => p.innerText.trim())
                .filter(t => t.length > 20)
                .join('\n\n');
        }

        return {
            titulo, autor, fecha, texto,
            medio: "Página|12",
            url: window.location.href
        };
    }
    """

    with sync_playwright() as p:
        # Lanzamos el navegador en modo headless
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
        )

        for url in urls:
            page = context.new_page()
            try:
                print(f"Procesando: {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_selector("article", timeout=15000)

                # Ejecución sincrónica del JS
                resultado = page.evaluate(JS_EXTRACTOR)
                data.append(resultado)
                print(f"✅ Capturado: {resultado['titulo'][:50]}...")

            except Exception as e:
                print(f"❌ Error en {url}: {e}")
            finally:
                page.close()

        browser.close()

    # Guardar los resultados en JSON
    with open("pagina12.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n🚀 Proceso exitoso. {len(data)} noticias en 'pagina12.json'.")

if __name__ == "__main__":
    extraer_pagina12()
