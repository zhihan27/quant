/* Streamlit 内置控件的中文操作文案；仅处理界面文字，不改输入值和表格数据。 */
(() => {
    // 脚本不占页面布局空间，避免在标题上方留下空白。
    const host = document.currentScript?.closest('[data-testid="stElementContainer"]');
    if (host) host.style.display = "none";
    const version = 1;
    const previous = window.__quantUiZhCn;
    if (previous?.version === version) {
        previous.refresh(document.body);
        return;
    }
    previous?.observer.disconnect();

    const messages = {
        "Main menu": "更多操作",
        "Settings": "设置",
        "About": "关于",
        "About Streamlit": "关于界面框架",
        "Print": "打印",
        "Record a screencast": "录制屏幕",
        "Record screen": "录制屏幕",
        "Start recording": "开始录制",
        "Stop recording": "停止录制",
        "Rerun": "重新运行",
        "Stop": "停止",
        "Running...": "正在运行…",
        "Clear cache": "清理缓存",
        "Clear cache?": "是否清理缓存？",
        "Get help": "查看帮助",
        "Report a bug": "反馈问题",
        "View app source": "查看应用源码",
        "Manage app": "管理应用",
        "Deploy": "发布",
        "Save": "保存",
        "Cancel": "取消",
        "Close": "关闭",
        "Done": "完成",
        "Confirm": "确认",
        "Apply": "应用",
        "Submit": "提交",
        "Reset": "重置",
        "Refresh": "刷新",
        "Retry": "重试",
        "OK": "确定",
        "Yes": "是",
        "No": "否",
        "Theme": "主题",
        "Choose app theme, colors and fonts": "选择界面主题、颜色和字体",
        "Light": "浅色",
        "Dark": "深色",
        "Use system setting": "跟随系统",
        "Use wide mode": "使用宽屏布局",
        "Run on save": "保存后自动运行",
        "Show app in wide mode": "以宽屏布局显示应用",
        "Automatically rerun script when the file changes": "文件修改后自动重新运行",
        "Expand sidebar": "展开侧栏",
        "Collapse sidebar": "收起侧栏",
        "App pages": "功能页面",
        "Skip to main content": "跳到主要内容",
        "Link to heading": "复制章节链接",
        "Open": "展开选项",
        "Select": "选择",
        "Select all": "全选",
        "Select All": "全选",
        "Clear all": "全部清除",
        "Clear All": "全部清除",
        "Clear value": "清除已选内容",
        "Clear selection": "清除选择",
        "No results": "没有匹配结果",
        "No options": "没有可选项",
        "Choose an option": "请选择",
        "Choose options": "请选择，可多选",
        "Selectbox options": "可选内容",
        "Multiselect options": "可多选内容",
        "Show password": "显示密码",
        "Hide password": "隐藏密码",
        "Increment": "增加",
        "Decrement": "减少",
        "Search": "搜索",
        "Search table": "搜索表格",
        "Search in table": "搜索表格内容",
        "Search columns": "搜索列",
        "Search...": "搜索…",
        "Search…": "搜索…",
        "Find": "查找",
        "Find in table": "在表格中查找",
        "Match case": "区分大小写",
        "Match whole word": "匹配完整词语",
        "Previous": "上一个",
        "Next": "下一个",
        "Download": "下载",
        "Download as CSV": "下载表格文件",
        "Download as PNG": "下载图片",
        "Download as SVG": "下载矢量图片",
        "Download as JSON": "下载数据文件",
        "Fullscreen": "全屏查看",
        "Close fullscreen": "退出全屏",
        "Enter fullscreen": "全屏查看",
        "Exit fullscreen": "退出全屏",
        "Expand": "展开",
        "Collapse": "收起",
        "Copy": "复制",
        "Copied!": "已复制",
        "Copy to clipboard": "复制到剪贴板",
        "Copy to Clipboard": "复制到剪贴板",
        "Copy column name": "复制列名",
        "Paste": "粘贴",
        "Cut": "剪切",
        "Undo": "撤销",
        "Redo": "重做",
        "Add row": "添加一行",
        "Delete row": "删除所选行",
        "Delete rows": "删除所选行",
        "Sort ascending": "升序排列",
        "Sort descending": "降序排列",
        "Clear sorting": "取消排序",
        "Hide column": "隐藏此列",
        "Show column": "显示此列",
        "Show all columns": "显示全部列",
        "Show/hide columns": "显示或隐藏列",
        "Pin column": "固定此列",
        "Unpin column": "取消固定",
        "Autosize": "自动调整宽度",
        "Autosize columns": "自动调整列宽",
        "Format": "显示格式",
        "Reset formatting": "恢复默认格式",
        "Data grid": "数据表格",
        "Dataframe toolbar": "表格工具栏",
        "Dataframe column menu": "列操作菜单",
        "Close Search": "关闭搜索",
        "Next Result": "下一个结果",
        "Previous Result": "上一个结果",
        "Type to search": "输入要搜索的内容",
        "No data": "暂无数据",
        "Show data": "查看数据",
        "View data": "查看数据",
        "View source": "查看图表配置",
        "View Source": "查看图表配置",
        "View Compiled Vega": "查看完整图表配置",
        "Open in Vega Editor": "在图表编辑器中打开",
        "Save as PNG": "保存为图片",
        "Save as SVG": "保存为矢量图片",
        "Click to show actions": "展开图表操作",
        "Download plot as a PNG": "下载图表图片",
        "Zoom": "框选放大",
        "Pan": "平移图表",
        "Box Select": "框选数据",
        "Lasso Select": "套索选择",
        "Zoom in": "放大",
        "Zoom out": "缩小",
        "Autoscale": "自动缩放",
        "Reset axes": "重置坐标轴",
        "Toggle Spike Lines": "切换辅助线",
        "Show closest data on hover": "悬停显示最近数据",
        "Compare data on hover": "悬停比较数据",
        "Previous month": "上个月",
        "Next month": "下个月",
        "Choose date": "选择日期",
        "Choose date range": "选择日期范围",
        "Clear date": "清除日期",
        "Clear dates": "清除日期",
        "Date range": "日期范围",
        "Quick select a date range": "快速选择日期范围",
        "January": "一月",
        "February": "二月",
        "March": "三月",
        "April": "四月",
        "May": "五月",
        "June": "六月",
        "July": "七月",
        "August": "八月",
        "September": "九月",
        "October": "十月",
        "November": "十一月",
        "December": "十二月",
        "Su": "日",
        "Mo": "一",
        "Tu": "二",
        "We": "三",
        "Th": "四",
        "Fr": "五",
        "Sa": "六",
        "Today": "今天",
        "Clear": "清除",
    };
    const attributes = ["aria-label", "aria-description", "title", "placeholder", "data-title"];
    const controlScope = [
        "button", "a", '[role="button"]', '[role="menu"]', '[role="menuitem"]',
        '[role="tooltip"]', '[role="dialog"]', '[data-baseweb="popover"]',
        '[data-testid="InputInstructions"]', '[data-testid="stSelectbox"]',
        '[data-testid="stMultiSelect"]', '[data-testid="stElementToolbar"]',
        ".vega-actions",
    ].join(",");
    const dataScope = 'script,style,pre,code,textarea,[contenteditable="true"],[role="gridcell"],[role="columnheader"],[role="option"]';

    function translate(value) {
        const key = value.trim();
        let replacement = Object.hasOwn(messages, key) ? messages[key] : null;
        if (!replacement && key.startsWith("Help for ")) {
            replacement = "查看说明：" + key.slice(9);
        }
        if (!replacement && /^Press (Enter|Ctrl\+Enter|⌘\+Enter) to (apply|submit)$/.test(key)) {
            replacement = key.replace(/^Press (.+) to (apply|submit)$/, (_, shortcut, action) =>
                "按 " + shortcut + (action === "submit" ? " 提交" : " 应用修改"));
        }
        if (!replacement && /^Remove .+/.test(key)) {
            replacement = "移除：" + key.slice(7);
        }
        return replacement ? value.replace(key, replacement) : value;
    }

    function refresh(root) {
        if (!root) return;
        const elements = root.nodeType === Node.ELEMENT_NODE
            ? [root, ...root.querySelectorAll("*")]
            : [root.parentElement];
        for (const element of elements) {
            if (!element || element.closest(dataScope)) continue;
            // 属性只包含辅助说明；不改 value、股票名称或表格单元格。
            for (const name of attributes) {
                const value = element.getAttribute(name);
                if (value !== null) {
                    const localized = translate(value);
                    if (localized !== value) element.setAttribute(name, localized);
                }
            }
            if (!element.closest(controlScope)) continue;
            for (const child of element.childNodes) {
                if (child.nodeType !== Node.TEXT_NODE) continue;
                const localized = translate(child.nodeValue);
                if (localized !== child.nodeValue) child.nodeValue = localized;
            }
        }
    }

    // 每页只保留一个监听器，合并同一帧的更新，覆盖重跑、弹出菜单和自动刷新片段。
    const pending = new Set();
    let scheduled = false;
    const observer = new MutationObserver(records => {
        for (const record of records) {
            if (record.type === "childList") {
                for (const node of record.addedNodes) pending.add(node);
            } else {
                pending.add(record.target);
            }
        }
        if (scheduled || !pending.size) return;
        scheduled = true;
        requestAnimationFrame(() => {
            scheduled = false;
            const roots = [...pending];
            pending.clear();
            for (const root of roots) {
                if (root.isConnected) refresh(root);
            }
        });
    });
    observer.observe(document.body, {
        subtree: true,
        childList: true,
        characterData: true,
        attributes: true,
        attributeFilter: attributes,
    });
    document.documentElement.lang = "zh-CN";
    window.__quantUiZhCn = {version, observer, refresh};
    refresh(document.body);
})();
