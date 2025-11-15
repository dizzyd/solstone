# Sunstone App Development Guide

**Complete guide for building apps in the `apps/` directory.**

Apps are the primary way to extend Sunstone's web interface (Convey). Each app is a self-contained module discovered automatically using **convention over configuration**—no base classes or manual registration required.

---

## Quick Start

Create a new app in three steps:

```bash
# 1. Create app directory (use underscores, not hyphens!)
mkdir apps/my_app

# 2. Create required files
touch apps/my_app/routes.py
touch apps/my_app/workspace.html

# 3. Add optional metadata
touch apps/my_app/app.json
```

**Minimal `routes.py`:**
```python
from flask import Blueprint, render_template

my_app_bp = Blueprint("app:my_app", __name__, url_prefix="/app/my_app")

@my_app_bp.route("/")
def index():
    return render_template("app.html", app="my_app")
```

**Minimal `workspace.html`:**
```html
<h1>Hello from My App!</h1>
```

**That's it!** Restart Convey and your app appears in the menu bar.

---

## Directory Structure

```
apps/my_app/
├── routes.py          # Required: Flask blueprint with routes
├── workspace.html     # Required: Main content template
├── app.json          # Optional: Metadata (icon, label, facet support)
├── hooks.py          # Optional: Dynamic submenu and badge logic
├── app_bar.html      # Optional: Bottom bar controls (forms, buttons)
└── background.html   # Optional: Background JavaScript service
```

### File Purposes

| File | Required | Purpose |
|------|----------|---------|
| `routes.py` | **Yes** | Flask blueprint with route handlers |
| `workspace.html` | **Yes** | Main app content (rendered in container) |
| `app.json` | No | Icon, label, facet support overrides |
| `hooks.py` | No | Submenu items and facet badge counts |
| `app_bar.html` | No | Bottom fixed bar for app controls |
| `background.html` | No | Background service (WebSocket listeners) |

---

## Naming Conventions

**Critical for auto-discovery:**

1. **App directory**: Use `snake_case` (e.g., `my_app`, **not** `my-app`)
2. **Blueprint variable**: Must be `{app_name}_bp` (e.g., `my_app_bp`)
3. **Blueprint name**: Must be `app:{app_name}` (e.g., `"app:my_app"`)
4. **URL prefix**: Convention is `/app/{app_name}` (e.g., `/app/my_app`)

**Why it matters**: The registry scans directories and imports `{app_name}_bp` from `apps.{app_name}.routes`. Incorrect naming prevents discovery.

---

## Required Files

### 1. `routes.py` - Flask Blueprint

Define all routes for your app using Flask blueprints.

**Pattern:**
```python
from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from convey import state
from convey.utils import error_response, success_response

# Blueprint variable MUST be {app_name}_bp
my_app_bp = Blueprint(
    "app:my_app",              # Blueprint name (use app:{name} pattern)
    __name__,
    url_prefix="/app/my_app"   # URL prefix
)

@my_app_bp.route("/")
def index():
    """Main view - render app container with workspace."""
    return render_template("app.html", app="my_app")

@my_app_bp.route("/data")
def get_data():
    """API endpoint returning JSON."""
    data = {"message": "Hello from API"}
    return jsonify(data)

@my_app_bp.route("/action", methods=["POST"])
def handle_action():
    """POST endpoint with error handling."""
    value = request.form.get("value")
    if not value:
        return error_response("Missing value", 400)

    # Process...
    return success_response({"result": value})
```

**Key Points:**
- All routes render `app.html` as the container, passing `app="my_app"`
- Use `url_for('app:my_app.index')` for internal links
- Access journal root via `state.journal_root` (always available)
- See [Flask Utilities](#flask-utilities) for helper functions

### 2. `workspace.html` - Main Content

The workspace template is included inside the app container (`app.html`).

**Pattern:**
```html
<style>
/* App-specific styles (scoped by class name) */
.my-app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}
</style>

<div class="my-app-container">
  <h1>My App</h1>

  <!-- Access template context -->
  <p>Selected facet: {{ selected_facet or 'All facets' }}</p>

  <!-- Render markdown using vendor library -->
  <div id="content"></div>

  <script src="{{ vendor_lib('marked') }}"></script>
  <script>
    const markdown = "# Hello\n\nThis is **markdown**.";
    document.getElementById('content').innerHTML = marked.parse(markdown);
  </script>
</div>
```

**Available Template Context:**
- `app` - Current app name
- `facets` - List of active facet dicts: `[{name, title, color, emoji}, ...]`
- `selected_facet` - Currently selected facet name (string or None)
- `app_registry` - Registry with all apps (usually not needed directly)
- `state.journal_root` - Path to journal directory
- Any variables passed from route handler via `render_template(...)`

**Vendor Libraries:**
- Use `{{ vendor_lib('marked') }}` for markdown rendering
- See `convey/static/vendor/VENDOR.md` for available libraries

---

## Optional Files

### 3. `app.json` - Metadata

Override default icon, label, and facet support.

**Format:**
```json
{
  "icon": "🏠",
  "label": "Home Dashboard",
  "facets": true
}
```

**Fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `icon` | string | "📦" | Emoji icon for menu bar |
| `label` | string | Title-cased name | Display label in menu |
| `facets` | boolean | `true` | Enable facet integration |

**Defaults:**
- Icon: "📦"
- Label: `app_name.replace("_", " ").title()` (e.g., "my_app" → "My App")
- Facets: `true` (facet pills shown, selection enabled)

**When to disable facets:** Set `"facets": false` for apps that don't use facet-based organization (e.g., system settings, dev tools).

### 4. `hooks.py` - Dynamic Logic

Provide submenu items and facet badge counts that update dynamically.

**Pattern:**
```python
from think.todo import get_todos

def get_submenu_items(facets, selected_facet):
    """Return submenu items for the menu-bar.

    Args:
        facets: List of active facet dicts [{name, title, color, emoji}, ...]
        selected_facet: Currently selected facet name (string or None)

    Returns:
        List of dicts with keys:
        - label: Display text (required)
        - path: URL path (required)
        - count: Optional badge count (int)
        - facet: Optional facet name for data-facet attribute
    """
    return [
        {"label": "Active", "path": "/app/my_app", "count": 5},
        {"label": "Archived", "path": "/app/my_app?status=archived"},
        {"label": "Work Items", "path": "/app/my_app", "facet": "work", "count": 3},
    ]

def get_facet_counts(facets, selected_facet):
    """Return badge counts for facet pills.

    Args:
        facets: List of active facet dicts
        selected_facet: Currently selected facet name (string or None)

    Returns:
        Dict mapping facet name to count, e.g.:
        {"work": 5, "personal": 3, "acme": 12}
    """
    counts = {}
    for facet in facets:
        todos = get_todos(date.today().strftime("%Y%m%d"), facet["name"])
        counts[facet["name"]] = len([t for t in todos if not t.get("completed")])
    return counts
```

**Submenu Items:**
- Appear below your app in the menu bar when expanded
- Optional `count` shows badge next to label
- Optional `facet` attribute enables facet selection on click (same behavior as facet pills)
- Without `facet`, items navigate directly to `path`

**Facet Counts:**
- Show badge on facet pills in the top bar
- Useful for showing counts per facet (e.g., pending todos, unread messages)

### 5. `app_bar.html` - Bottom Bar Controls

Fixed bottom bar for forms, buttons, date pickers, search boxes.

**Pattern:**
```html
<form method="POST" action="{{ url_for('app:my_app.add_item') }}" class="app-bar-form">
  <input type="text" name="text" placeholder="Add new item..." autofocus>
  <button type="submit">Add</button>
</form>

<style>
.app-bar-form {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.app-bar-form input {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
}

.app-bar-form button {
  padding: 0.5rem 1rem;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
</style>
```

**Key Points:**
- App bar is fixed to bottom when present
- Page body gets `has-app-bar` class (adjusts content margin)
- Only rendered when app provides this template
- Great for persistent input controls across views

### 6. `background.html` - Background Service

JavaScript service that runs globally, even when app is not active.

**Pattern:**
```html
{# Background service runs on all pages #}
window.AppServices.register('my_app', {

  initialize() {
    console.log('[My App Service] Initialized');

    // Check WebSocket availability
    if (!window.appEvents) {
      console.warn('[My App Service] WebSocket not available');
      return;
    }

    // Listen to specific tract
    window.appEvents.listen('cortex', this.handleCortexEvent.bind(this));

    // Or listen to all events
    window.appEvents.listen('*', this.handleAnyEvent.bind(this));
  },

  handleCortexEvent(msg) {
    if (msg.event === 'agent_complete') {
      // Show persistent notification card
      window.AppServices.notifications.show({
        app: 'my_app',
        icon: '🤖',
        title: 'Agent Complete',
        message: `${msg.agent} finished processing`,
        action: '/app/my_app',
        facet: msg.facet,  // Auto-select facet on click
        badge: 1,
        autoDismiss: 10000  // Dismiss after 10 seconds
      });

      // Update badge count
      window.AppServices.updateBadge('my_app', msg.facet, 5);
    }
  },

  handleAnyEvent(msg) {
    console.log('[My App Service] Event:', msg.tract, msg);
  },

  isCurrentApp() {
    // Check if my_app is currently active
    const menuItem = document.querySelector('.menu-item[data-app="my_app"]');
    return menuItem && menuItem.classList.contains('current');
  }

});
```

**AppServices API:**

**Core Methods:**
- `AppServices.register(appName, service)` - Register background service
- `AppServices.updateBadge(appName, facetName, count)` - Update facet badge
- `AppServices.updateSubmenu(appName, items)` - Update submenu items

**Notification Methods:**
- `AppServices.notifications.show(options)` - Show persistent notification card
- `AppServices.notifications.dismiss(id)` - Dismiss specific notification
- `AppServices.notifications.dismissApp(appName)` - Dismiss all for app
- `AppServices.notifications.dismissAll()` - Dismiss all notifications
- `AppServices.notifications.count()` - Get active notification count
- `AppServices.notifications.update(id, options)` - Update existing notification

**Notification Options:**
```javascript
{
  app: 'my_app',          // App name (required)
  icon: '📬',             // Emoji icon (optional)
  title: 'New Message',   // Title (required)
  message: 'You have...', // Message body (optional)
  action: '/app/inbox',   // Click action URL (optional)
  facet: 'work',          // Auto-select facet on click (optional)
  badge: 5,               // Badge count (optional)
  dismissible: true,      // Show X button (default: true)
  autoDismiss: 10000      // Auto-dismiss ms (optional)
}
```

**WebSocket Events (`window.appEvents`):**
- `listen(tract, callback)` - Listen to specific tract ('cortex', 'indexer', 'observe', etc.)
- `listen('*', callback)` - Listen to all events
- Messages have structure: `{tract: 'cortex', event: 'agent_complete', ...data}`
- See **CALLOSUM.md** for event protocol details

---

## Flask Utilities

Available helper functions from `convey.utils`:

### Route Helpers

```python
from convey.utils import error_response, success_response, parse_pagination_params

# Standard error response
return error_response("Invalid input", 400)
# Returns: (jsonify({"error": "Invalid input"}), 400)

# Standard success response
return success_response({"agent_id": "123"})
# Returns: (jsonify({"success": True, "agent_id": "123"}), 200)

# Parse pagination parameters with validation
limit, offset = parse_pagination_params(default_limit=20, max_limit=100)
```

### Date Navigation

```python
from convey.utils import adjacent_days, format_date

# Get previous and next day directories
prev_day, next_day = adjacent_days(state.journal_root, "20250114")
# Returns: ("20250113", "20250115") or (None, None) if not found

# Format date for display
formatted = format_date("20250114")
# Returns: "Wednesday January 14th"
```

### Agent Spawning

```python
from convey.utils import spawn_agent

# Spawn a Cortex agent
agent_id = spawn_agent(
    prompt="Summarize today's transcripts",
    persona="summarizer",
    backend="openai",
    config={"facet": "work", "model": "gpt-4"}
)
# Returns: agent_id (timestamp-based string)
```

### JSON Utilities

```python
from convey.utils import load_json, save_json

# Load JSON file (returns None on error)
data = load_json("/path/to/file.json")

# Save JSON file (returns True on success)
success = save_json("/path/to/file.json", {"key": "value"})
```

---

## Think Module Integration

Available functions from the `think` module:

### Facets

```python
from think.facets import get_facets

# Get all facets
facets = get_facets()
# Returns: {"work": {title, color, emoji, disabled}, ...}
```

### Todos

```python
from think.todo import get_todos, TodoChecklist

# Get todos for a day and facet
todos = get_todos("20250114", "work")
# Returns: [{"text": "...", "completed": False, "index": 1}, ...]

# Manage todo checklist
checklist = TodoChecklist.load("20250114", "work")
checklist.append_entry("New todo item")
checklist.mark_done(1, guard="- [ ] New todo item")
checklist.mark_undone(1, guard="- [x] New todo item")
checklist.remove_entry(1, guard="- [x] New todo item")
```

### Messages

```python
from think.messages import get_unread_count, get_messages

# Get unread message count
count = get_unread_count()

# Get messages (with pagination)
messages = get_messages(limit=20, offset=0, status="active")
```

### Entities

```python
from think.entities import get_entities

# Get entities for a facet
entities = get_entities(facet="work")
# Returns: [{"name": "...", "type": "person", "facet": "work"}, ...]
```

See **JOURNAL.md** for journal structure, **CORTEX.md** for agent system, and **CALLOSUM.md** for event protocol.

---

## JavaScript APIs

### Global Variables

Available on all app pages:

```javascript
// Facet data (array of facet objects)
window.facetsData
// [{name: "work", title: "Work", color: "#3b82f6", emoji: "💼"}, ...]

// Selected facet from server
window.selectedFacetFromServer
// "work" or null

// Facet badge counts for current app
window.appFacetCounts
// {"work": 5, "personal": 3}
```

### Facet Selection

Listen for facet changes in your workspace:

```javascript
window.addEventListener('facet.switch', (e) => {
  const { facet, facetData } = e.detail;
  console.log('Facet switched to:', facet);  // "work" or null
  console.log('Facet data:', facetData);     // {name, title, color, emoji}

  // Reload content for new facet
  if (facet) {
    loadFacetContent(facet);
  } else {
    loadAllContent();
  }
});

// Programmatically switch facet
selectFacet('work');  // Select specific facet
selectFacet(null);    // Switch to all-facet mode
```

**Facet Modes:**
- **all-facet mode**: `facet = null`, show content from all facets
- **specific-facet mode**: `facet = "work"`, show only that facet's content
- Selection persisted via cookie, synchronized across pills and submenu items

### WebSocket Events

Available via `window.appEvents` (loaded from `websocket.js`):

```javascript
// Listen to specific tract
window.appEvents.listen('cortex', (msg) => {
  console.log('Cortex event:', msg);
  // msg = {tract: 'cortex', event: 'agent_complete', agent: 'summarizer', ...}
});

// Listen to all events
window.appEvents.listen('*', (msg) => {
  console.log('Any event:', msg.tract, msg);
});
```

**Common tracts:** `cortex`, `indexer`, `observe`, `task`

See **CALLOSUM.md** for complete event protocol.

---

## CSS Styling

### CSS Variables

Dynamic variables based on selected facet:

```css
:root {
  --facet-color: #3b82f6;      /* Selected facet color */
  --facet-bg: #3b82f61a;       /* 10% opacity background */
  --facet-border: #3b82f6;     /* Border color */
}

/* Use in your styles */
.my-element {
  background: var(--facet-bg);
  border: 1px solid var(--facet-border);
  color: var(--facet-color);
}
```

Variables update automatically when facet selection changes.

### App-Specific Styles

**Best practice:** Scope styles using a unique class:

```html
<style>
/* Scoped to .my-app-container */
.my-app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.my-app-container h1 {
  font-size: 2rem;
  color: var(--facet-color);
}
</style>

<div class="my-app-container">
  <h1>My App</h1>
</div>
```

### Global Styles

Main stylesheet `convey/static/app.css` provides base components. Review it for available classes and patterns.

---

## Common Patterns

### Date-Based Navigation

```python
from datetime import date
from convey.utils import adjacent_days, format_date
from convey import state

@my_app_bp.route("/<day>")
def view_day(day: str):
    # Validate date format
    if not re.match(r"\d{8}", day):
        return "", 404

    # Get navigation
    prev_day, next_day = adjacent_days(state.journal_root, day)
    today = date.today().strftime("%Y%m%d")

    # Load day data...

    return render_template(
        "app.html",
        app="my_app",
        day=day,
        prev_day=prev_day,
        next_day=next_day,
        today=today,
        title=format_date(day)
    )
```

### AJAX Endpoints

```python
@my_app_bp.route("/api/data", methods=["POST"])
def api_data():
    # Parse JSON payload
    payload = request.get_json(silent=True) or {}
    value = payload.get("value")

    if not value:
        return error_response("Missing value", 400)

    # Process...
    result = process_value(value)

    return success_response({"result": result})
```

### Form Handling with Flash Messages

```python
from flask import flash, redirect, url_for

@my_app_bp.route("/add", methods=["POST"])
def add_item():
    text = request.form.get("text", "").strip()

    if not text:
        flash("Cannot add empty item", "error")
        return redirect(url_for("app:my_app.index"))

    # Add item...
    flash("Item added successfully", "success")
    return redirect(url_for("app:my_app.index"))
```

### Facet-Aware Queries

```python
from flask import request

@my_app_bp.route("/")
def index():
    # Get selected facet from cookie
    selected_facet = request.cookies.get("selectedFacet")

    if selected_facet:
        # Load data for specific facet
        items = load_items(facet=selected_facet)
    else:
        # Load data from all facets
        items = load_all_items()

    return render_template("app.html", app="my_app", items=items)
```

---

## Debugging Tips

### Check Discovery

```bash
# Start Convey with debug logging
FLASK_DEBUG=1 convey

# Look for log lines:
# "Discovered app: my_app"
# "Registered blueprint: app:my_app"
```

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| App not discovered | Missing `routes.py` or `workspace.html` | Ensure both files exist |
| Blueprint not found | Wrong variable name | Use `{app_name}_bp` exactly |
| Import error | Blueprint name mismatch | Use `"app:{app_name}"` exactly |
| Hyphens in name | Directory uses hyphens | Rename to use underscores |
| Routes don't work | URL prefix mismatch | Check `url_prefix` matches pattern |

### Logging

```python
import logging
from flask import current_app

@my_app_bp.route("/")
def index():
    current_app.logger.info("Rendering my_app index")
    current_app.logger.debug("Selected facet: %s", selected_facet)
    # ...
```

---

## Best Practices

1. **Use underscores** in directory names (`my_app`, not `my-app`)
2. **Scope CSS** with unique class names to avoid conflicts
3. **Validate input** on all POST endpoints (use `error_response`)
4. **Check facet selection** when loading facet-specific data
5. **Use state.journal_root** for journal path (always available)
6. **Provide hooks** if app has submenu or facet counts
7. **Handle errors gracefully** with flash messages or JSON errors
8. **Test facet switching** to ensure content updates correctly
9. **Use background services** for WebSocket event handling
10. **Follow Flask patterns** for blueprints, url_for, etc.

---

## Example Apps

Study these reference implementations:

- **`apps/home/`** - Minimal app with background service
- **`apps/todos/`** - Full-featured with date navigation, forms, AJAX
- **`apps/inbox/`** - Submenu with badges via hooks
- **`apps/dev/`** - Custom styling and notification testing
- **`apps/tokens/`** - App bar with form controls

---

## Additional Resources

- **CLAUDE.md** - Project development guidelines and standards
- **JOURNAL.md** - Journal directory structure and data organization
- **CORTEX.md** - Agent system architecture and spawning agents
- **CALLOSUM.md** - Message bus protocol and WebSocket events
- **CRUMBS.md** - Transcript format specification

For Flask documentation, see [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
