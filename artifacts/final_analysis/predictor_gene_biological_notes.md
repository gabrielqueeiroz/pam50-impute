# Predictor gene biological notes (leve)

Notas qualitativas com base em conhecimento estabelecido da literatura PAM50 / biologia mamária. **Sem** enriquecimento funcional e **sem** consulta a bases externas nesta análise.

## MKI67 (frequência como preditor: 1427)

Antígeno Ki-67; marcador clássico de proliferação celular. Associado a ciclo celular / fração de crescimento tumoral; não é receptor hormonal nem HER2. Papel: índice proliferativo em mama.

## NDC80 (frequência como preditor: 1376)

Componente do complexo cinetocoro NDC80; essencial para segregação cromossómica na mitose. Associado a ciclo celular/proliferação. Não é receptor hormonal nem HER2.

## UBE2T (frequência como preditor: 1341)

Enzima conjugadora de ubiquitina; envolvida em reparo de DNA (via FANCL) e turnover proteico. Ligada a proliferação/manutenção genómica; não é receptor hormonal nem HER2.

## CEP55 (frequência como preditor: 1340)

Proteína de centrossomo/citocinese; regula abscisão celular. Fortemente ligada a ciclo celular e proliferação. Não é receptor hormonal nem HER2.

## PTTG1 (frequência como preditor: 1301)

Securina (pituitary tumor-transforming 1); regula separação de cromátides-irmãs. Associada a ciclo celular, proliferação e instabilidade genómica. Não é receptor hormonal nem HER2.

## KIF2C (frequência como preditor: 1289)

Cinesina (MCAK); despolimeriza microtúbulos no cinetocoro. Função mitótica / ciclo celular / proliferação. Não é receptor hormonal nem HER2.

## UBE2C (frequência como preditor: 1260)

Ubiquitina-conjugase do ciclo anáfase (APC/C); degradação de ciclina B e saída da mitose. Marcador de proliferação/ciclo. Não é receptor hormonal nem HER2.

## EXO1 (frequência como preditor: 1254)

Exonucleases 1; reparo de DNA (mismatch/recombinação). Associada a manutenção genómica e frequentemente coexpressa com assinaturas proliferativas. Não é receptor hormonal nem HER2.

## RRM2 (frequência como preditor: 1170)

Subunidade da ribonucleotídeo redutase; síntese de dNTPs para replicação do DNA. Ligada a ciclo S / proliferação. Não é receptor hormonal nem HER2.

## CCNE1 (frequência como preditor: 1154)

Ciclina E1; transição G1/S. Driver clássico de ciclo celular e proliferação; amplificação relatada em subtipos agressivos. Não é receptor hormonal nem HER2 (embora relevante em Basal/HG).

---

## Existe padrão biológico evidente?

**Sim, com cautela:** os 10 preditores mais frequentes são predominantemente genes de **proliferação / ciclo celular / mitose / replicação e reparo de DNA** (MKI67, NDC80, CEP55, PTTG1, KIF2C, UBE2C, UBE2T, EXO1, RRM2, CCNE1). Não dominam nesta lista os receptores hormonais clássicos (ESR1, PGR) nem ERBB2/HER2 — esses aparecem mais como **alvos** difíceis de imputar do que como preditores top (ver `gene_difficulty_report.md`).

**Interpretação cautelosa:** o padrão é compatível com a estrutura de correlação do PAM50 (módulo proliferativo coeso), não com uma descoberta biológica nova. A seleção Pearson+RFE tende a escolher genes altamente correlacionados com o alvo; no METABRIC isso frequentemente cai no bloco proliferativo. **Não** se deve inferir causalidade, especificidade terapêutica ou validação clínica a partir destas frequências.

Se o critério for um padrão além de 'proliferação', **não há evidência clara** nos resultados de um segundo eixo dominante (p.ex. invasão ou diferenciação luminal) entre os top-10 preditores.
