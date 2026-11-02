#!/usr/bin/env bash
# Build the anonymous submission and the non-anonymous version of the paper for every venue.
#
#     cd paper && ./build_all.sh
#
# All versions share body.tex, appendix.tex and tables/*.tex; each venue file only sets the
# template, the authors and the citation style. A venue file is anonymous unless \FINAL is
# defined, which this script does for the non-anonymous builds. PDFs are written to build/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

# latex <source .tex basename> <jobname> <final: 0|1>
latex() {
    local src=$1 job=$2 final=$3 pre=""
    [ "$final" = 1 ] && pre='\def\FINAL{1}'
    for pass in 1 2; do
        pdflatex -interaction=nonstopmode -halt-on-error -jobname="$job" "$pre\\input{$src}" > "build/$job.console" 2>&1 \
            || { echo "FAILED: $job (see build/$job.console)"; tail -20 "build/$job.console"; exit 1; }
        [ "$pass" = 1 ] && { bibtex "$job" > "build/$job.bibtex" 2>&1 || true; }
    done
    pdflatex -interaction=nonstopmode -halt-on-error -jobname="$job" "$pre\\input{$src}" > "build/$job.console" 2>&1
    rm -f "build/$job.console" "build/$job.bibtex"
}

# venue  source        anonymous output           non-anonymous output
while read -r venue src anon final; do
    latex "$src" "${src}"       0; cp "${src}.pdf"       "build/$anon"
    latex "$src" "${src}_final" 1; cp "${src}_final.pdf" "build/$final"
    echo "$venue: build/$anon, build/$final"
done <<'EOF'
ARR/ACL paper_acl   ARR_ACL_submission.pdf    ACL_camera_ready_draft.pdf
ICML    paper_icml  ICML_submission.pdf       ICML_preprint.pdf
IJCAI   paper_ijcai IJCAI_submission.pdf      IJCAI_camera_ready_draft.pdf
COLM    paper_colm  COLM_submission.pdf       COLM_preprint.pdf
TMLR    paper_tmlr  TMLR_submission.pdf       TMLR_preprint.pdf
EOF

# IJCAI appendix: separate supplementary file (needs paper_ijcai*.aux from the builds above)
latex paper_ijcai_supp paper_ijcai_supp       0; cp paper_ijcai_supp.pdf       build/IJCAI_supplementary_submission.pdf
latex paper_ijcai_supp paper_ijcai_supp_final 1; cp paper_ijcai_supp_final.pdf build/IJCAI_supplementary_camera_ready_draft.pdf
echo "IJCAI supplementary: build/IJCAI_supplementary_submission.pdf"

# arXiv preprint (ACL layout with authors)
latex paper_arxiv paper_arxiv 0; cp paper_arxiv.pdf build/arXiv_preprint.pdf
echo "arXiv: build/arXiv_preprint.pdf"

rm -f ./*_final.{aux,bbl,blg,log,out,pdf} ./*.blg
