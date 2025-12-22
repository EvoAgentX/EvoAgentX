from typing import Dict, Any, Optional
import json
import tkinter as tk
from tkinter import ttk

class WorkFlowJSONEditorGUI:
    """GUI JSON Editor GUI based on tkinter"""
    
    def __init__(self, json_data: Dict[str, Any]):
        self.json_data = json_data
        self.result = None
        self.root = None
        
    def edit_json(self) -> Optional[Dict[str, Any]]:
        """start the json editor and return the modified data"""
        try:
            import tkinter as tk
            from tkinter import ttk, scrolledtext
        except ImportError:
            print("⚠️  tkinter is not available, use the text editor")
            return self._edit_json_text()
        
        self.root = tk.Tk()
        self.root.title("WorkFlow JSON Editor")
        self.root.geometry("800x600")
        
        # create the main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # configure the grid weight
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # title
        title_label = ttk.Label(main_frame, text="Edit WorkFlow JSON Structure", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # left button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.N), padx=(0, 10))
        
        # buttons
        ttk.Button(button_frame, text="📝 Format", command=self._format_json).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="✅ Validate", command=self._validate_json).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔄 Reset", command=self._reset_json).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="📋 Copy", command=self._copy_json).pack(fill=tk.X, pady=2)
        
        ttk.Separator(button_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # quick operation buttons
        ttk.Label(button_frame, text="Quick Operations:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(button_frame, text="➕ Add Node", command=self._add_node_quick).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔗 Add Edge", command=self._add_edge_quick).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="📄 Template", command=self._insert_template).pack(fill=tk.X, pady=2)
        
        # right text editor area
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # text editor
        self.text_area = scrolledtext.ScrolledText(
            text_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=30,
            font=("Consolas", 10)
        )
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # insert json data
        self.text_area.insert(tk.END, json.dumps(self.json_data, indent=2, ensure_ascii=False))
        
        # bottom button frame
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # confirm and cancel buttons
        ttk.Button(bottom_frame, text="💾 Save and Close", command=self._save_and_close).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(bottom_frame, text="❌ Cancel", command=self._cancel).pack(side=tk.LEFT, padx=(0, 5))
        
        # status label
        self.status_label = ttk.Label(bottom_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.RIGHT)
        
        # start the gui
        self.root.mainloop()
        return self.result
    
    def _format_json(self):
        """format the json"""
        try:
            text = self.text_area.get(1.0, tk.END)
            data = json.loads(text)
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, formatted)
            self.status_label.config(text="✅ Formatting completed", foreground="green")
        except json.JSONDecodeError as e:
            self.status_label.config(text=f"❌ JSON format error: {e}", foreground="red")
    
    def _validate_json(self):
        """validate the json"""
        try:
            text = self.text_area.get(1.0, tk.END)
            data = json.loads(text)
            
            # validate the workflow structure
            if not isinstance(data, dict):
                raise ValueError("The root node must be a dictionary")
            
            if 'nodes' not in data or not isinstance(data['nodes'], list):
                raise ValueError("Must contain nodes array")
            
            node_names = set()
            for node in data['nodes']:
                if not isinstance(node, dict) or 'name' not in node:
                    raise ValueError("Each node must contain name field")
                
                name = node['name']
                if name in node_names:
                    raise ValueError(f"Node name duplicate: {name}")
                node_names.add(name)
            
            # validate the edges
            if 'edges' in data:
                for edge in data['edges']:
                    if not isinstance(edge, dict):
                        continue
                    source = edge.get('source')
                    target = edge.get('target')
                    if source and source not in node_names:
                        raise ValueError(f"The source node of the edge does not exist: {source}")
                    if target and target not in node_names:
                        raise ValueError(f"The target node of the edge does not exist: {target}")
            
            self.status_label.config(text="✅ JSON structure is valid", foreground="green")
            
        except (json.JSONDecodeError, ValueError) as e:
            self.status_label.config(text=f"❌ Validation failed: {e}", foreground="red")
    
    def _reset_json(self):
        """reset the json"""
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, json.dumps(self.json_data, indent=2, ensure_ascii=False))
        self.status_label.config(text="🔄 Reset", foreground="blue")
    
    def _copy_json(self):
        """copy the json to the clipboard"""
        try:
            text = self.text_area.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="📋 Copied to clipboard", foreground="blue")
        except Exception as e:
            self.status_label.config(text=f"❌ Copy failed: {e}", foreground="red")
    
    def _add_node_quick(self):
        """quick add node"""
        try:
            import tkinter.simpledialog as sd
            
            name = sd.askstring("Add Node", "Node name:")
            if not name:
                return
            
            desc = sd.askstring("Add Node", "Node description:")
            if not desc:
                desc = f"The description of the node {name}"
            
            node_template = {
                "class_name": "WorkFlowNode",
                "name": name,
                "description": desc,
                "inputs": [],
                "outputs": [],
                "agents": [],
                "status": "pending"
            }
            
            # get the current json
            current_text = self.text_area.get(1.0, tk.END)
            try:
                data = json.loads(current_text)
                data.setdefault('nodes', []).append(node_template)
                
                # update the text area
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, json.dumps(data, indent=2, ensure_ascii=False))
                self.status_label.config(text=f"✅ Added node: {name}", foreground="green")
                
            except json.JSONDecodeError:
                self.status_label.config(text="❌ Current JSON format error, cannot add node", foreground="red")
                
        except ImportError:
            self.status_label.config(text="❌ Cannot use dialog", foreground="red")
    
    def _add_edge_quick(self):
        """quick add edge"""
        try:
            import tkinter.simpledialog as sd
            
            # get the current node list
            current_text = self.text_area.get(1.0, tk.END)
            try:
                data = json.loads(current_text)
                nodes = data.get('nodes', [])
                node_names = [node.get('name') for node in nodes if node.get('name')]
                
                if len(node_names) < 2:
                    self.status_label.config(text="❌ At least 2 nodes are required to add edge", foreground="red")
                    return
                
                source = sd.askstring("Add Edge", f"Source node (optional: {', '.join(node_names)}):")
                if not source or source not in node_names:
                    self.status_label.config(text="❌ Source node invalid", foreground="red")
                    return
                
                target = sd.askstring("Add Edge", f"Target node (optional: {', '.join(node_names)}):")
                if not target or target not in node_names:
                    self.status_label.config(text="❌ Target node invalid", foreground="red")
                    return
                
                edge_template = {
                    "class_name": "WorkFlowEdge",
                    "source": source,
                    "target": target,
                    "priority": 0
                }
                
                data.setdefault('edges', []).append(edge_template)
                
                # update the text area
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, json.dumps(data, indent=2, ensure_ascii=False))
                self.status_label.config(text=f"✅ Added edge: {source} -> {target}", foreground="green")
                
            except json.JSONDecodeError:
                self.status_label.config(text="❌ Current JSON format error, cannot add edge", foreground="red")
                
        except ImportError:
            self.status_label.config(text="❌ Cannot use dialog", foreground="red")
    
    def _insert_template(self):
        """insert template"""
        templates = {
            "Simple Node": {
                "class_name": "WorkFlowNode",
                "name": "new_node",
                "description": "New node description",
                "inputs": [{"class_name": "Parameter", "name": "input1", "type": "string", "description": "Input parameter", "required": True}],
                "outputs": [{"class_name": "Parameter", "name": "output1", "type": "string", "description": "Output parameter", "required": True}],
                "agents": [],
                "status": "pending"
            },
            "CustomizeAgent": {
                "name": "my_agent",
                "description": "Customize Agent",
                "inputs": [{"name": "input1", "type": "string", "description": "Input", "required": True}],
                "outputs": [{"name": "output1", "type": "string", "description": "Output", "required": True}],
                "prompt": "Process input: {input1}",
                "parse_mode": "str"
            }
        }
        
        # create the template selection window
        template_window = tk.Toplevel(self.root)
        template_window.title("Select Template")
        template_window.geometry("400x300")
        
        ttk.Label(template_window, text="Select the template to insert:").pack(pady=10)
        
        template_listbox = tk.Listbox(template_window)
        template_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for template_name in templates.keys():
            template_listbox.insert(tk.END, template_name)
        
        def insert_selected():
            selection = template_listbox.curselection()
            if selection:
                template_name = template_listbox.get(selection[0])
                template_json = json.dumps(templates[template_name], indent=2, ensure_ascii=False)
                self.text_area.insert(tk.INSERT, f"\n{template_json}\n")
                self.status_label.config(text=f"✅ Inserted template: {template_name}", foreground="green")
                template_window.destroy()
        
        ttk.Button(template_window, text="Insert", command=insert_selected).pack(pady=10)
        ttk.Button(template_window, text="Cancel", command=template_window.destroy).pack()
    
    def _save_and_close(self):
        """save and close"""
        try:
            text = self.text_area.get(1.0, tk.END)
            self.result = json.loads(text)
            self.root.destroy()
        except json.JSONDecodeError as e:
            self.status_label.config(text=f"❌ JSON format error: {e}", foreground="red")
    
    def _cancel(self):
        """cancel"""
        self.result = None
        self.root.destroy()
    
    def _edit_json_text(self) -> Optional[Dict[str, Any]]:
        """use the text editor to edit the json (backup solution)"""
        import tempfile
        import subprocess
        import os
        
        # create the temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)
            temp_file = f.name
        
        try:
            print(f"📝 Opening file editor: {temp_file}")
            print("💡 Please save the file and close the editor after editing")
            
            # select the editor based on the operating system
            if os.name == 'nt':  # Windows
                subprocess.run(['notepad', temp_file])
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(['nano', temp_file])
            
            # read the edited file
            with open(temp_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            return result
            
        except Exception as e:
            print(f"❌ Editor opening failed: {e}")
            return None
        finally:
            # clean up the temporary file
            os.unlink(temp_file)