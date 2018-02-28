FROM quay.io/openai/universe.world-of-bits:0.20.11-4556541
COPY --chown=root:root changedFiles/run.py          /app/universe-envs/world-of-bits/bin/
RUN chmod 755 /app/universe-envs/world-of-bits/bin/run.py
COPY --chown=root:root changedFiles/core.js         /app/universe-envs/world-of-bits/static/core/
COPY --chown=root:root changedFiles/simple-algebra.html /app/universe-envs/world-of-bits/static/miniwob/
COPY --chown=root:root changedFiles/visual-addition.html /app/universe-envs/world-of-bits/static/miniwob/
COPY --chown=root:root changedFiles/number-checkboxes.html /app/universe-envs/world-of-bits/static/miniwob/
COPY --chown=root:root changedFiles/moving-items.html /app/universe-envs/world-of-bits/static/miniwob/
COPY --chown=root:root changedFiles/click-test.html /app/universe-envs/world-of-bits/static/miniwob/